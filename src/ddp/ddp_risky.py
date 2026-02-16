"""
Offline risky-model DDP solver.

Implements a dynamic corporate finance model with risky debt and optimal
capital investment. The solver uses a loop-within-a-loop equilibrium:
1. Inner loop: firm optimization for a fixed bond-price schedule.
2. Outer loop: lender bond-price fixed point from endogenous default risk.

Transition probabilities are estimated from observed (z, z_next_main) pairs
in the input flat dataset. No internal shock simulation is performed.
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple

import tensorflow as tf

from src.economy.data import DatasetBundle
from src.economy.parameters import EconomicParams, convert_to_tf
from src.ddp.ddp_config import (
    DDPGridConfig,
    estimate_transition_matrix_from_dataset,
    extract_bounds_from_metadata,
    initialize_markov_process,
)
from src.economy import logic


class RiskyModelDDP:
    """
    Solves a dynamic corporate finance model with risky debt and endogenous
    interest rates using Discrete Dynamic Programming (DDP).

    Overview
    --------
    This model solves for the optimal investment and financing policies of a firm
    that can finance investment using risky debt. The interest rate is endogenously
    determined by a risk-neutral lender based on the firm's default probability.

    Key Features:
        1. 3D State Space: (z: Productivity, k: Capital, b: Existing Debt).
        2. Endogenous Risk: Bond prices q(z, k', b') are solved via fixed-point iteration.
        3. Default: Shareholders choose to default if Equity Value < 0.
        4. Scalability: Uses tf.map_fn to handle large Z-grids without OOM errors.

    Attributes:
        params (EconomicParams): Container for economic parameters.
        grid_config (DDPGridConfig): Grid discretization settings.
        beta (tf.Tensor): Discount factor.
        z_grid (tf.Tensor): Productivity grid. Shape: (nz,).
        prob_matrix (tf.Tensor): Transition probability matrix. Shape: (nz, nz).
        k_grid (tf.Tensor): Capital grid. Shape: (nk,).
        b_grid (tf.Tensor): Bond/Debt grid. Shape: (nb,).
        nz (int): Size of productivity grid.
        nk (int): Size of capital grid.
        nb (int): Size of bond grid.
        static_flows (tf.Tensor): Precomputed 5D tensor of cash flows invariant to bond prices.
    """

    def __init__(
        self, 
        params: EconomicParams,
        shock_params: Optional[Any] = None,
        grid_config: Optional[DDPGridConfig] = None,
        *,
        dataset: Optional[Dict[str, tf.Tensor]] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        delta: Optional[float] = None,
        reward_matrix_fn: Optional[Callable[..., tf.Tensor]] = None,
    ):
        """
        Initialize the model, grids, and precompute static cash flows.

        Args:
            params: Economic parameters.
            dataset: Flattened dataset with at least {'z', 'z_next_main'}.
            dataset_metadata: Metadata containing canonical bounds.
            grid_config: Numerical grid settings.
            delta: Optional depreciation override for grid construction.
            reward_matrix_fn: Optional custom reward matrix builder.
        """
        self.params = params
        self.shock_params = shock_params
        self.grid_config = grid_config or DDPGridConfig()
        self.beta = tf.constant(1 / (1 + params.r_rate), dtype=tf.float32)
        self.reward_matrix_fn = reward_matrix_fn

        if dataset is not None and dataset_metadata is not None:
            self.dataset = dataset
            self.dataset_metadata = dataset_metadata
            bounds = extract_bounds_from_metadata(dataset_metadata)
            delta_val = float(params.delta if delta is None else delta)

            # --- 1. Generate Grids ---
            k_grid_np, _, b_grid_np = self.grid_config.generate_grids(bounds, delta=delta_val)
            z_grid_np, prob_matrix_np = estimate_transition_matrix_from_dataset(
                dataset,
                bounds,
                z_size=self.grid_config.z_size,
            )
        else:
            # Backward-compatible legacy path.
            if shock_params is None:
                raise ValueError(
                    "Offline path requires dataset + dataset_metadata. "
                    "Legacy path requires shock_params."
                )
            z_grid_np, prob_matrix_np = initialize_markov_process(shock_params, self.grid_config.z_size)
            k_grid_np = self.grid_config.generate_capital_grid(params)
            b_grid_np = self.grid_config.generate_bond_grid(
                params,
                k_max=k_grid_np.max(),
                z_min=z_grid_np.min(),
            )
            self.dataset = None
            self.dataset_metadata = None

        # --- 2. Convert to TensorFlow Constants ---
        self.z_grid, self.prob_matrix, self.k_grid, self.b_grid = convert_to_tf(
            z_grid_np, prob_matrix_np, k_grid_np, b_grid_np
        )

        # --- 3. Store Dimensions ---
        self.nz = self.grid_config.z_size
        self.nk = len(k_grid_np)
        self.nb = len(b_grid_np)

        # --- 4. Precompute Static Flows ---
        self.static_flows = self._build_static_cash_flows()

    @classmethod
    def from_dataset_bundle(
        cls,
        params: EconomicParams,
        bundle: DatasetBundle,
        *,
        grid_config: Optional[DDPGridConfig] = None,
        delta: Optional[float] = None,
        reward_matrix_fn: Optional[Callable[..., tf.Tensor]] = None,
    ) -> "RiskyModelDDP":
        """
        Convenience constructor for metadata-first offline workflows.
        """
        return cls(
            params,
            dataset=bundle.data,
            dataset_metadata=bundle.metadata,
            grid_config=grid_config,
            delta=delta,
            reward_matrix_fn=reward_matrix_fn,
        )

    def _build_static_cash_flows(self) -> tf.Tensor:
        """
        Constructs the 5D tensor of cash flows that do NOT depend on bond price 'q'.

        Formula:
            Flow = (1-tax)*Profit - Investment - AdjCosts - OldDebt + TaxShield_FaceValue

        Returns:
            tf.Tensor: 5D Tensor (nz, nk, nk, nb, nb) containing static cash flows.
        """
        params = self.params

        # Reshape grids for 4D broadcasting
        # We are inside a logic that will be wrapped by tf.map_fn, so we only
        # care about the 4D structure (nk, nk, nb, nb).
        # Shape Target: (nk_curr, nk_next, b_curr, b_next)
        k_curr = tf.reshape(self.k_grid, [self.nk, 1, 1, 1])
        k_next = tf.reshape(self.k_grid, [1, self.nk, 1, 1])
        b_curr = tf.reshape(self.b_grid, [1, 1, self.nb, 1])
        b_next = tf.reshape(self.b_grid, [1, 1, 1, self.nb])

        # 1. Investment and Adjustment Costs (z-invariant)
        # Shape: (nk, nk, 1, 1)
        # Note: Use small temperature for near-hard gates in discrete DP
        investment = logic.compute_investment(k_curr, k_next, params)
        adj_costs = logic.adjustment_costs(
            k_curr, k_next, params,
            temperature=1e-6,  # Near-hard gate for discrete DP
            logit_clip=20.0
        )

        # Define function to process a single Z
        def process_z_static(z_val):
            """Calculates the (nk, nk, nb, nb) static cash flow matrix for a given z scalar."""
            # 1. Operating Profit: Broadcasts to (nk, 1, 1, 1)
            profit = logic.production_function(k_curr, z_val, params)

            # 2. Tax Shield (Face Value portion): Broadcasts to (1, 1, 1, nb)
            tax_shield_static = (params.tax * b_next) / (1 + params.r_rate)

            # 3. Combine Components (Profit - Inv - Costs - Repay + Shield)
            # Note: We calculate 'Cash Flow' excluding the NEW DEBT term (which depends on price)
            return (1 - params.tax) * profit - investment - adj_costs - b_curr + tax_shield_static

        # Execute map_fn to iterate over the productivity grid (z_grid)
        # This replaces the Python loop, processing slices in parallel if possible.
        # tf.map_fn stacks the (nk, nk, nb, nb) results into (nz, nk, nk, nb, nb)
        return tf.map_fn(
            process_z_static,
            self.z_grid,
            fn_output_signature=tf.float32,
            parallel_iterations=10
        )

    @tf.function
    def _compute_reward_matrix(self, bond_price_schedule: tf.Tensor) -> tf.Tensor:
        """
        Computes the 5-dimensional reward matrix (Dividends) for the firm.

        This method combines pre-computed static flows with dynamic financing flows
        derived from the current `bond_price_schedule`. It uses `tf.map_fn` to
        process each productivity state (z) sequentially (or in batches), avoiding
        the memory overhead of broadcasting the full 5D tensors at once.

        Args:
            bond_price_schedule (tf.Tensor): The equilibrium market price q(z, k', b')
                for risky debt. Shape: (nz, nk, nb).

        Returns:
            tf.Tensor: The 5D reward matrix representing Div(z, k, k', b, b').
                Shape: (nz, nk, nk, nb, nb).
        """
        if self.reward_matrix_fn is not None:
            return self.reward_matrix_fn(
                bond_price_schedule=bond_price_schedule,
                z_grid=self.z_grid,
                k_grid=self.k_grid,
                b_grid=self.b_grid,
                prob_matrix=self.prob_matrix,
                params=self.params,
                temperature=1e-6,
                logit_clip=20.0,
            )

        params = self.params

        # --- 1. Pre-calculation & Reshaping for Broadcasting ---

        # Net Proceeds Factor:
        # Represents the fraction of bond issuance kept after tax effects.
        # Derived from: q * b' - (tax * q * b') / (1 + r)
        net_proceeds_factor = 1.0 - (params.tax / (1 + params.r_rate))

        # Reshape global grids to align with the 4D structure inside the loop:
        # Target Structure inside loop: (k_curr, k_next, b_curr, b_next)

        # b_next (Next Debt) applies to Axis 3
        # Shape: (nb,) -> (1, 1, 1, nb)
        b_next = tf.reshape(self.b_grid, [1, 1, 1, self.nb])

        # k_safe (Current Capital) applies to Axis 0. Used for scaling injection costs.
        # Shape: (nk,) -> (nk, 1, 1, 1)
        k_safe = tf.reshape(tf.maximum(self.k_grid, 1e-6), [self.nk, 1, 1, 1])

        # --- 2. Define Logic for Single Z-State ---

        def process_z_reward(inputs):
            """
            Inner function to calculate dividends for a single productivity state z.

            Args:
                inputs: A tuple containing slices for the current z:
                    1. div_static: Static cash flows. Shape: (nk, nk, nb, nb).
                    2. q_slice: Bond prices q(k', b'). Shape: (nk, nb).
            """
            div_static, q_slice = inputs

            # A. Broadcast Bond Prices
            # q_slice corresponds to Future State (k_next, b_next).
            # We must map these to Axis 1 (k_next) and Axis 3 (b_next) of the 4D structure.
            # Input: (nk, nb) -> Target: (1, nk, 1, nb)
            q_broadcast = tf.reshape(q_slice, [1, self.nk, 1, self.nb])

            # B. Calculate Financing Proceeds
            # Proceeds = q(k', b') * b' * factor
            # Operation: (1, nk, 1, nb) * (1, 1, 1, nb) -> Result: (1, nk, 1, nb)
            financing_flow = (q_broadcast * b_next) * net_proceeds_factor

            # C. Total Dividends
            # Add static flows (depends on k, k', b, b') and financing (depends on k', b')
            # Operation: (nk, nk, nb, nb) + (1, nk, 1, nb) -> Result: (nk, nk, nb, nb)
            dividends = div_static + financing_flow

            # D. Apply Equity Injection Costs
            # Use shared logic: Cost = is_negative * (Fixed + Linear*|div|)
            # NOTE: indicator is evaluated on normalized e/k per report_brief.md lines 432-433
            # Note: Use small temperature for near-hard gates in discrete DP
            injection_cost = logic.external_financing_cost(
                dividends, k_safe, params,
                temperature=1e-6,  # Near-hard gate for discrete DP
                logit_clip=20.0
            )

            return dividends - injection_cost

        # --- 3. Execute Loop over Z (Map Fn) ---

        # We 'zip' the static flows and bond prices together into a tuple.
        # tf.map_fn slices Dimension 0 (nz) of both tensors simultaneously.
        # Input 1: static_flows (nz, nk, nk, nb, nb)
        # Input 2: bond_price   (nz, nk, nb)
        # Output: (nz, nk, nk, nb, nb)
        return tf.map_fn(
            process_z_reward,
            elems=(self.static_flows, bond_price_schedule),
            fn_output_signature=tf.float32,
            parallel_iterations=10
        )

    @tf.function
    def _compute_bellman_step_with_policy(
        self,
        v_curr: tf.Tensor,
        reward_matrix: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Bellman step that returns both values and argmax policy indices.

        Returns:
            v_next: Updated value function (nz, nk, nb).
            pol_k_idx: Optimal k' index for each state (nz, nk, nb).
            pol_b_idx: Optimal b' index for each state (nz, nk, nb).
        """
        v_flat = tf.reshape(v_curr, [self.nz, -1])
        ev_flat = tf.matmul(self.prob_matrix, v_flat)
        ev = tf.reshape(ev_flat, [self.nz, self.nk, self.nb])
        cv = self.beta * ev

        value_spec = tf.TensorSpec((self.nk, self.nb), tf.float32)
        index_spec = tf.TensorSpec((self.nk, self.nb), tf.int32)

        def process_z_bellman(inputs):
            r_slice, cv_slice = inputs

            cv_broad = tf.reshape(cv_slice, [1, self.nk, 1, self.nb])
            rhs = r_slice + cv_broad

            # Reorder to (k_curr, b_curr, k_next, b_next), then flatten action dims.
            rhs_ordered = tf.transpose(rhs, perm=[0, 2, 1, 3])
            rhs_flat = tf.reshape(rhs_ordered, [self.nk, self.nb, -1])

            best_linear_idx = tf.argmax(rhs_flat, axis=-1, output_type=tf.int32)
            v_unconstrained = tf.reduce_max(rhs_flat, axis=-1)
            v_state = tf.maximum(v_unconstrained, 0.0)

            best_k_idx = best_linear_idx // self.nb
            best_b_idx = best_linear_idx % self.nb
            return v_state, best_k_idx, best_b_idx

        v_next, pol_k_idx, pol_b_idx = tf.map_fn(
            process_z_bellman,
            (reward_matrix, cv),
            fn_output_signature=(value_spec, index_spec, index_spec),
            parallel_iterations=10
        )
        return v_next, pol_k_idx, pol_b_idx

    @tf.function
    def _compute_bellman_step(self, v_curr: tf.Tensor, reward_matrix: tf.Tensor) -> tf.Tensor:
        """
        Bellman step returning only updated values.

        This is used by value-only callers (for example, some unit tests).
        """
        v_next, _, _ = self._compute_bellman_step_with_policy(v_curr, reward_matrix)
        return v_next

    @tf.function
    def _compute_bond_price(self, v_next: tf.Tensor) -> tf.Tensor:
        """
        Calculates the equilibrium bond unit price schedule q(z, k', b').

        This function implements the lender's Zero-Profit Condition under Strategic
        Default. The lender observes the firm's continuation value function
        V(z', k', b') and anticipates default whenever V < 0.

        Args:
            v_next (tf.Tensor): The firm's continuation value function V(z', k', b')
                from the previous iteration. Used to identify the 'Default Set'.
                Shape: (nz, nk, nb).

        Returns:
            tf.Tensor: The market price per unit of face value. Bounded in [0, 1/(1+r)].
                Shape: (nz, nk, nb).
        """
        params = self.params

        # 1. Determine Default States

        # Default: Value is 0 (or less)
        default_bool = tf.math.less_equal(v_next, 0.0)
        is_default = tf.cast(default_bool, tf.float32)

        # 2. Calculate Lender Payoff (Solvent vs Default)
        # is_default tells us which regime we are in.
        
        # We need to broadcast z_next and k_next for the payoff function
        # z_grid: (nz,) -> (nz, 1, 1)
        z_next_broad = tf.reshape(self.z_grid, [self.nz, 1, 1])
        # k_grid: (nk,) -> (1, nk, 1) 
        k_next_broad = tf.reshape(self.k_grid, [1, self.nk, 1])
        # b_grid: (nb,) -> (1, 1, self.nb)
        b_next_broad = tf.reshape(self.b_grid, [1, 1, self.nb])
        
        # Calculate Payoff using shared logic
        # Note: logic.compute_lender_payoff handles Recovery, Min(Rec, Debt), and Smoothing
        future_payoff = logic.compute_lender_payoff(
            k_next_broad, 
            b_next_broad, 
            z_next_broad, 
            is_default, 
            params
        )

        # 4. Expectation & Discounting
        # E[Payoff | z] = ProbMatrix @ FuturePayoff
        payoff_flat = tf.reshape(future_payoff, [self.nz, -1])
        # Matrix Mult: (nz, nz) @ (nz, nk*nb) -> (nz, nk*nb)
        expected_payoff_flat = tf.matmul(self.prob_matrix, payoff_flat)
        expected_payoff = tf.reshape(expected_payoff_flat, [self.nz, self.nk, self.nb])

        total_bond_value = expected_payoff / (1 + params.r_rate)

        # 5. Convert to Unit Price: q = 1 / (1 + r_tilde)
        # Use divide_no_nan and pin b'<=0 cases to risk-free price.
        # Under borrowing-only grids, this only affects b'=0.
        bond_unit_price = tf.math.divide_no_nan(total_bond_value, b_next_broad)
        rf_price = 1.0 / (1.0 + params.r_rate)
        bond_unit_price = tf.where(
            b_next_broad <= 0.0,
            tf.cast(rf_price, bond_unit_price.dtype),
            bond_unit_price
        )

        # Numerical guardrail: enforce no-arbitrage price bounds.
        bond_unit_price = tf.clip_by_value(
            bond_unit_price,
            clip_value_min=0.0,
            clip_value_max=tf.cast(rf_price, bond_unit_price.dtype),
        )

        return bond_unit_price

    @tf.function
    def _evaluate_policy_indices(
        self,
        policy_k_idx: tf.Tensor,
        policy_b_idx: tf.Tensor,
        reward_matrix: tf.Tensor,
        v_init: tf.Tensor,
        steps: int = 300,
    ) -> tf.Tensor:
        """
        Howard-style policy evaluation for fixed risky-model policy indices.

        Limited-liability option (default to zero) remains active during
        evaluation via value = max(policy_value, 0).
        """
        policy_linear_idx = policy_k_idx * self.nb + policy_b_idx
        output_spec = tf.TensorSpec((self.nk, self.nb), tf.float32)

        def gather_reward(inputs):
            r_slice, idx_slice = inputs
            r_ordered = tf.transpose(r_slice, perm=[0, 2, 1, 3])  # (k, b, k', b')
            r_flat = tf.reshape(r_ordered, [self.nk, self.nb, -1])
            return tf.gather(r_flat, idx_slice, axis=2, batch_dims=2)

        reward_policy = tf.map_fn(
            gather_reward,
            (reward_matrix, policy_linear_idx),
            fn_output_signature=output_spec,
            parallel_iterations=10,
        )

        value = v_init
        for _ in range(steps):
            v_flat = tf.reshape(value, [self.nz, -1])
            ev_flat = tf.matmul(self.prob_matrix, v_flat)
            ev_grid = tf.reshape(ev_flat, [self.nz, self.nk, self.nb])

            def gather_ev(inputs):
                ev_slice, idx_slice = inputs
                ev_slice_flat = tf.reshape(ev_slice, [-1])
                return tf.gather(ev_slice_flat, idx_slice)

            ev_choice = tf.map_fn(
                gather_ev,
                (ev_grid, policy_linear_idx),
                fn_output_signature=output_spec,
                parallel_iterations=10,
            )
            value = tf.maximum(reward_policy + self.beta * ev_choice, 0.0)

        return value

    def solve_risky(
        self,
        tol: float = 1e-4,
        max_iter: int = 50,
        damping_weight: float = 1.0,
        inner_solver: Literal["vfi", "pfi"] = "vfi",
        inner_solver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Solve risky-model equilibrium with configurable inner firm solver.

        Args:
            tol: Outer-loop price convergence tolerance.
            max_iter: Maximum outer-loop iterations.
            damping_weight: Price update weight in (0, 1].
            inner_solver: "vfi" or "pfi" for the inner firm problem.
            inner_solver_kwargs: Optional kwargs passed to inner solver.
        """
        if inner_solver not in {"vfi", "pfi"}:
            raise ValueError(f"inner_solver must be 'vfi' or 'pfi'. Got '{inner_solver}'.")
        if not (0.0 < damping_weight <= 1.0):
            raise ValueError(f"damping_weight must be in (0, 1]. Got {damping_weight}.")

        inner_solver_kwargs = inner_solver_kwargs or {}
        print(f"Starting Risky DDP Solver (inner={inner_solver.upper()}, damping={damping_weight})...")

        v_star = None
        policy = None
        rf_price = 1.0 / (1.0 + self.params.r_rate)
        q_current = tf.ones((self.nz, self.nk, self.nb), dtype=tf.float32) * rf_price

        for _ in range(max_iter):
            reward_matrix = self._compute_reward_matrix(q_current)

            if inner_solver == "vfi":
                v_star, policy = self.solve_vfi(reward_matrix, **inner_solver_kwargs)
            else:
                v_star, policy = self.solve_pfi(reward_matrix, **inner_solver_kwargs)

            q_new = self._compute_bond_price(v_star)
            diff = float(tf.reduce_max(tf.abs(q_new - q_current)).numpy())
            if diff < tol:
                print("Strategic Equilibrium Converged!")
                return v_star, policy, q_new

            q_current = q_current + damping_weight * (q_new - q_current)
            q_current = tf.clip_by_value(q_current, 0.0, tf.cast(rf_price, tf.float32))

        print("Warning: Strategic Loop reached max iterations.")
        return v_star, policy, q_current

    def solve_risky_vfi(
        self,
        tol: float = 1e-4,
        max_iter: int = 50,
        damping_weight: float = 1.0,
        inner_vfi_tol: float = 1e-5,
        inner_vfi_max_iter: int = 2000,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Risky-model equilibrium with inner VFI.
        """
        return self.solve_risky(
            tol=tol,
            max_iter=max_iter,
            damping_weight=damping_weight,
            inner_solver="vfi",
            inner_solver_kwargs={"tol": inner_vfi_tol, "max_iter": inner_vfi_max_iter},
        )

    def solve_risky_pfi(
        self,
        tol: float = 1e-4,
        max_iter: int = 50,
        damping_weight: float = 1.0,
        inner_pfi_max_iter: int = 200,
        inner_pfi_eval_steps: int = 400,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Risky-model equilibrium with inner Howard-style policy iteration.
        """
        return self.solve_risky(
            tol=tol,
            max_iter=max_iter,
            damping_weight=damping_weight,
            inner_solver="pfi",
            inner_solver_kwargs={
                "max_iter": inner_pfi_max_iter,
                "eval_steps": inner_pfi_eval_steps,
            },
        )

    @staticmethod
    def _argmax_2d(tensor_4d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Finds the (row, col) indices that maximize the last two dimensions.

        Args:
            tensor_4d (tf.Tensor): Input tensor of shape (..., rows, cols).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Best row indices, Best col indices.
        """
        # 1. Capture dynamic shapes
        shape = tf.shape(tensor_4d)
        # We only need the size of the LAST dimension (the 'width' or columns)
        n_cols_out = shape[-1]

        # 2. Flatten the search space (last two dims) into one
        # Shape: (..., n_rows * n_cols)
        flat_search_space = tf.reshape(tensor_4d, tf.concat([shape[:-2], [-1]], axis=0))

        # 3. Find linear index of the maximum
        best_linear_idx = tf.argmax(flat_search_space, axis=-1, output_type=tf.int32)

        # 4. Decode (Row = Index // Width, Col = Index % Width)
        best_rows = best_linear_idx // n_cols_out
        best_cols = best_linear_idx % n_cols_out

        return best_rows, best_cols

    def _extract_policy(
        self, reward_matrix: tf.Tensor, v_final: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extracts the optimal policy functions k'(z, k, b) and b'(z, k, b).

        This helper method performs one final pass over the Bellman equation to
        recover the *arguments* that maximize the value function (argmax), rather
        than just the value itself.

        Args:
            reward_matrix (tf.Tensor): Payoff matrix (nz, nk, nk, nb, nb).
            v_final (tf.Tensor): Converged value function V(z, k, b).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                - policy_k: Optimal capital choice k'(z, k, b).
                - policy_b: Optimal debt choice b'(z, k, b).
        """
        # 1. Calculate the Continuation Value
        v_flat = tf.reshape(v_final, [self.nz, -1])
        ev_flat = tf.matmul(self.prob_matrix, v_flat)
        ev = tf.reshape(ev_flat, [self.nz, self.nk, self.nb])
        cv = self.beta * ev  # Shape: (nz, nk, nb)

        pol_k_list = []
        pol_b_list = []

        for i_z in range(self.nz):
            # Current Reward: (k_curr, k_next, b_curr, b_next)
            r_slice = reward_matrix[i_z]

            # Continuation Value: Broadcast to (1, nk, 1, nb)
            cv_broad = tf.reshape(cv[i_z], [1, self.nk, 1, self.nb])

            # Total Value RHS: (k_curr, k_next, b_curr, b_next)
            rhs = r_slice + cv_broad

            # --- ROBUST OPTIMIZATION STEP ---

            # 1. Define Axis Constants (The "Named Axis" Pattern)
            # Original Layout: (k_curr, k_next, b_curr, b_next)
            axis_k_curr, axis_k_next, axis_b_curr, axis_b_next = 0, 1, 2, 3

            # 2. Transpose with explicit intent
            # "We want States (k_curr, b_curr) on left, Actions (k_next, b_next) on right"
            rhs_ordered = tf.transpose(
                rhs, perm=[axis_k_curr, axis_b_curr, axis_k_next, axis_b_next]
            )

            # C. Runtime Shape Assertion
            # Ensures dimensions are exactly where we think they are.
            tf.debugging.assert_shapes(
                [(rhs_ordered, (self.nk, self.nb, self.nk, self.nb))],
                message="Permutation Error: Dimensions are not aligned!",
            )

            # D. Get Optimal Indices (Encapsulated Logic)
            best_k_idx, best_b_idx = self._argmax_2d(rhs_ordered)

            # E. Retrieve actual values from grids
            pol_k_list.append(tf.gather(self.k_grid, best_k_idx))
            pol_b_list.append(tf.gather(self.b_grid, best_b_idx))

        # Stack z-slices back together
        return tf.stack(pol_k_list, axis=0), tf.stack(pol_b_list, axis=0)

    def solve_vfi(
        self, reward_matrix: tf.Tensor, tol: float = 1e-5, max_iter: int = 2000
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Iterates the Bellman Operator until convergence (Fixed Point).

        Args:
            reward_matrix (tf.Tensor): Precomputed payoffs.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum iterations.

        Returns:
            Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
                - v_star: Optimal value function.
                - policy: Tuple of optimal (k, b) choices.
        """
        v_curr = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.float32)
        pol_k_idx = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.int32)
        pol_b_idx = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.int32)

        for iteration in range(max_iter):
            v_next, pol_k_idx, pol_b_idx = self._compute_bellman_step_with_policy(
                v_curr, reward_matrix
            )
            diff = float(tf.reduce_max(tf.abs(v_next - v_curr)).numpy())
            if diff < tol:
                print(f"    VFI Converged: {iteration} iters, Diff={diff:.6f}")
                return v_next, (
                    tf.gather(self.k_grid, pol_k_idx),
                    tf.gather(self.b_grid, pol_b_idx),
                )
            v_curr = v_next

        print("    Warning: VFI did not converge.")
        return v_curr, (
            tf.gather(self.k_grid, pol_k_idx),
            tf.gather(self.b_grid, pol_b_idx),
        )

    def solve_pfi(
        self,
        reward_matrix: tf.Tensor,
        max_iter: int = 200,
        eval_steps: int = 400,
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Howard-style Policy Function Iteration for the risky-model inner problem.
        """
        policy_k_idx = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.int32)
        policy_b_idx = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.int32)
        v_curr = tf.zeros((self.nz, self.nk, self.nb), dtype=tf.float32)

        for iteration in range(max_iter):
            v_eval = self._evaluate_policy_indices(
                policy_k_idx, policy_b_idx, reward_matrix, v_curr, steps=eval_steps
            )
            v_greedy, new_k_idx, new_b_idx = self._compute_bellman_step_with_policy(
                v_eval, reward_matrix
            )

            k_changed = tf.reduce_any(tf.not_equal(new_k_idx, policy_k_idx))
            b_changed = tf.reduce_any(tf.not_equal(new_b_idx, policy_b_idx))
            policy_changed = bool(k_changed.numpy() or b_changed.numpy())
            if not policy_changed:
                print(f"    PFI Converged: {iteration + 1} policy updates.")
                return v_greedy, (
                    tf.gather(self.k_grid, new_k_idx),
                    tf.gather(self.b_grid, new_b_idx),
                )

            policy_k_idx = new_k_idx
            policy_b_idx = new_b_idx
            v_curr = v_greedy

        print("    Warning: PFI did not converge.")
        return v_curr, (
            tf.gather(self.k_grid, policy_k_idx),
            tf.gather(self.b_grid, policy_b_idx),
        )
