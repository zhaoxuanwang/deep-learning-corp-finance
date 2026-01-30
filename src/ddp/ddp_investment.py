"""
ddp_investment.py

Implements a Discrete Dynamic Programming (DDP) solver using TensorFlow.
This approach leverages GPU acceleration for Value Function Iteration (VFI)
and Policy Function Iteration (PFI).
"""

from typing import Tuple

import tensorflow as tf

from src.economy.parameters import EconomicParams, ShockParams, convert_to_tf
from src.ddp.ddp_config import DDPGridConfig, initialize_markov_process
from src.economy import logic
from typing import Optional


class InvestmentModelDDP:
    """
    Solves the firm optimal investment problem in dynamic models
    using Discrete Dynamic Programming (DDP)
    accelerated by TensorFlow.

    Attributes:
        params (EconomicParams): The economic configuration.
        grid_config (DDPGridConfig): Grid discretization settings.
        beta (tf.Tensor): Discount factor.
        z_grid (tf.Tensor): Productivity state grid (nz,).
        prob_matrix (tf.Tensor): Transition probability matrix (nz, nz).
        k_grid (tf.Tensor): Capital stock grid (nk,).
        nz (int): Number of productivity states.
        nk (int): Number of capital grid points.
        reward_matrix (tf.Tensor): Precomputed payoff matrix of shape (nz, nk, nk).
    """

    def __init__(
        self, 
        params: EconomicParams, 
        shock_params: ShockParams,
        grid_config: Optional[DDPGridConfig] = None
    ):
        """
        Initializes the model, generates grids, and precomputes rewards.

        Args:
            params (EconomicParams): The economic parameters.
            shock_params (ShockParams): The shock parameters.
            grid_config (DDPGridConfig): Grid settings (uses defaults if None).
        """
        self.params = params
        self.shock_params = shock_params
        self.grid_config = grid_config or DDPGridConfig()
        self.beta = tf.constant(1 / (1 + params.r_rate), dtype=tf.float32)

        # Generate grids using grid_config
        z_grid_np, prob_matrix_np = initialize_markov_process(shock_params, self.grid_config.z_size)
        k_grid_np = self.grid_config.generate_capital_grid(params)

        # Convert to TensorFlow Constants
        self.z_grid, self.prob_matrix, self.k_grid = convert_to_tf(
            z_grid_np, prob_matrix_np, k_grid_np
        )

        # Store dimensions
        self.nz = self.grid_config.z_size
        self.nk = len(k_grid_np)

        # Precompute the Reward Matrix
        self.reward_matrix = self._compute_reward_matrix()

    def _compute_reward_matrix(self) -> tf.Tensor:
        """
        Computes the 3D Reward Matrix R(z, k, k').

        This matrix represents the instantaneous payoff for every possible
        combination of current state (z, k) and choice (k').

        Returns:
            tf.Tensor: A 3D tensor of shape (nz, nk, nk).
                       Axis 0: Current productivity z
                       Axis 1: Current capital k
                       Axis 2: Next period capital k'
        """
        # Reshape for broadcasting
        # z: (nz, 1, 1)
        z_mesh = tf.reshape(self.z_grid, [self.nz, 1, 1])
        # k: (1, nk, 1)
        k_mesh = tf.reshape(self.k_grid, [1, self.nk, 1])
        # k': (1, 1, nk)
        k_next_mesh = tf.reshape(self.k_grid, [1, 1, self.nk])

        # --- Economic Logic ---
        
        # 1. Calculate Reward Components using shared 'logic' module
        # Note: logic functions support broadcasting.
        # k_mesh: (1, nk, 1) -> Current K
        # k_next_mesh: (1, 1, nk) -> Next K
        # z_mesh: (nz, 1, 1) -> Current Z
        
        # We can directly use the compute_cash_flow_basic function which wraps
        # Profit - I - Costs
        # Note: Use small temperature for near-hard gates in discrete DP
        net_cash_flow = logic.compute_cash_flow_basic(
            k_mesh, k_next_mesh, z_mesh, self.params,
            temperature=1e-6,  # Near-hard gate for discrete DP
            logit_clip=20.0
        )

        return net_cash_flow

    @tf.function
    def bellman_step(self, v_guess: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the Bellman Operator T(V) once.

        T(V)(z, k) = max_{k'} [ R(z, k, k') + beta * E[V(z', k') | z] ]

        Args:
            v_guess (tf.Tensor): The current value function guess of shape (nz, nk).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - v_new: Updated value function of shape (nz, nk).
                - policy_idx: Indices of optimal k' for each state, shape (nz, nk).
        """
        # 1. Expected Continuation Value: E[V(z', k')]
        # Matrix multiply: (nz, nz) @ (nz, nk) -> (nz, nk)
        expected_v = tf.matmul(self.prob_matrix, v_guess)

        # 2. Expand for broadcasting: (nz, nk) -> (nz, 1, nk)
        # Expand axis 1: E[V(z', k')] depends on chosen k' (axis 2)
        # but is independent of current k (axis 1).
        # This step aligns E[V] with the 'next capital' axis of the reward matrix
        expected_v_expanded = tf.expand_dims(expected_v, axis=1)

        # 3. Construct RHS: R + beta * E[V]
        # Shape: (nz, nk, nk)
        rhs = self.reward_matrix + self.beta * expected_v_expanded

        # 4. Maximization
        # v_new contains the max value across axis 2 (k')
        v_new = tf.reduce_max(rhs, axis=2)
        # policy_idx contains the index of that max value
        policy_idx = tf.argmax(rhs, axis=2)

        return v_new, policy_idx

    @tf.function
    def _evaluate_policy(self, policy_idx: tf.Tensor, v_init: tf.Tensor, steps: int = 100) -> tf.Tensor:
        """
        Policy Evaluation: Computes V_h for a fixed policy index h.

        Iterates T_h(V) = R_h + beta * P * V repeatedly to compute
        the infinite horizon value of a specific policy.

        Args:
            policy_idx (tf.Tensor): Indices of the fixed policy k'(z, k).
            v_init (tf.Tensor): Initial guess for value function.
            steps (int): Number of iterations for policy evaluation.

        Returns:
            tf.Tensor: The approximated value function for the given policy.
        """
        # 1. Extract Reward R(z, k) implied by policy h(z, k)
        # We need to grab one value per (z, k) pair from the 3rd dimension (k').
        # batch_dims=2 tells TF: For each (z, k) location, look at the index in policy_idx,
        # and pick the corresponding value from the axis 2 of reward_matrix.
        # Input Shapes: reward_matrix (nz, nk, nk), policy_idx (nz, nk)
        # Output Shape: (nz, nk)
        reward_policy = tf.gather(self.reward_matrix, policy_idx, axis=2, batch_dims=2)

        value = v_init

        # Calculate discounted PV over k-periods
        for _ in range(steps):
            # 2. Expected Continuation Value: E[V(z',k') | z]
            expected_v_grid = tf.matmul(self.prob_matrix, value)

            # 3. Select specific k_next.
            # We align the 'z' dimension (Axis 0) using batch_dims=1.
            # Inside each 'z', we have a row of future values (over k').
            # We use indices from policy_idx to select from these columns (Axis 1).
            # Input: expected_v_grid (nz, nk'), indices (nz, nk) -> Output: (nz, nk)
            expected_v_choice = tf.gather(expected_v_grid, policy_idx, axis=1, batch_dims=1)

            # 4. Update value
            value = reward_policy + self.beta * expected_v_choice

        return value

    def solve_invest_vfi(
        self, tol: float = 1e-6, max_iter: int = 1000
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Solves the model using Value Function Iteration (VFI).

        VFI is stable and reliable but converges linearly (slow).

        Args:
            tol (float): Convergence tolerance for the sup norm.
            max_iter (int): Maximum number of iterations.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - v_star: Optimal value function (nz, nk).
                - policy_star: Optimal capital choices (nz, nk).
        """
        v_current = tf.zeros((self.nz, self.nk), dtype=tf.float32)
        policy_idx = tf.zeros((self.nz, self.nk), dtype=tf.int64)

        i = 0
        error = tol + 1.0

        while error > tol and i < max_iter:
            v_next, policy_idx = self.bellman_step(v_current)
            error = tf.reduce_max(tf.abs(v_next - v_current))
            v_current = v_next
            i += 1

        if i >= max_iter:
            print(f"VFI Warning: Max iter {max_iter} reached. Error: {error:.2e}")
        else:
            print(f"VFI Converged in {i} steps. Error: {error:.2e}")

        # Map indices to actual capital values
        # Shape of k_grid: (nk,)
        # Shape of policy_idx (nz,nk)
        # Output: (nz,nk) filled with values from k_grid according to policy_idx
        policy_k = tf.gather(self.k_grid, policy_idx)
        return v_current, policy_k

    def solve_invest_pfi(
        self, max_iter: int = 100, eval_steps: int = 200
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Solves the model using Policy Function Iteration (PFI).

        Also known as Howard's Improvement Algorithm. Generally converges
        faster (quadratic rates) than VFI.

        Args:
            max_iter (int): Maximum number of policy improvement steps.
            eval_steps (int): Number of steps for partial policy evaluation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - v_star: Optimal value function (nz, nk).
                - policy_star: Optimal capital choices (nz, nk).
        """
        # Initialize with a zero policy
        policy_idx = tf.zeros((self.nz, self.nk), dtype=tf.int32)
        v_current = tf.zeros((self.nz, self.nk), dtype=tf.float32)

        i = 0
        policy_stable = False

        while not policy_stable and i < max_iter:
            # A. Policy Evaluation (Get precise V for current policy)
            v_eval = self._evaluate_policy(policy_idx, v_current, steps=eval_steps)

            # B. Policy Improvement (Greedy Step)
            v_greedy, new_policy_idx = self.bellman_step(v_eval)
            new_policy_idx = tf.cast(new_policy_idx, tf.int32)

            # C. Check Stability (Has the policy changed?)
            diff = tf.reduce_sum(tf.abs(new_policy_idx - policy_idx))

            if diff == 0:
                policy_stable = True
                print(f"PFI Converged in {i + 1} steps.")

            policy_idx = new_policy_idx
            v_current = v_greedy
            i += 1

        # Map indices to actual capital values
        policy_k = tf.gather(self.k_grid, policy_idx)
        return v_current, policy_k