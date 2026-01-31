"""
src/trainers/stopping.py

Convergence and stopping criteria for training loops.

Implements the hierarchical stopping logic from report_brief.md:
1. Gatekeeper (Annealing): No early stopping before N_anneal
2. Method-specific stopping rules (post-annealing)

References:
    report_brief.md: Convergence and Stopping Criteria
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Any
from collections import deque

from src.trainers.config import (
    DEFAULT_PATIENCE,
    DEFAULT_LR_EPSILON,
    DEFAULT_LR_WINDOW,
    DEFAULT_MA_WINDOW,
    DEFAULT_ER_EPSILON,
    DEFAULT_BR_CRITIC_EPSILON,
    DEFAULT_BR_ACTOR_EPSILON,
    DEFAULT_DIVISION_EPSILON
)

logger = logging.getLogger(__name__)


@dataclass
class StoppingState:
    """
    Tracks state for convergence checking.

    Attributes:
        patience_counter: Consecutive checks meeting convergence criteria
        loss_history: Recent loss values for moving average
        best_loss: Best loss seen so far (for improvement tracking)
    """
    patience_counter: int = 0
    loss_history: List[float] = None
    best_loss: float = float('inf')

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


class ConvergenceChecker:
    """
    Checks convergence criteria for training.

    Implements the hierarchical stopping logic:
    1. Gatekeeper: If step < n_anneal, ignore all early stopping
    2. Method-specific criteria (LR, ER, BR)

    """

    def __init__(
        self,
        method: str,
        n_anneal: int,
        patience: int = DEFAULT_PATIENCE,
        lr_epsilon: float = DEFAULT_LR_EPSILON,
        lr_window: int = DEFAULT_LR_WINDOW,
        er_epsilon: float = DEFAULT_ER_EPSILON,
        br_critic_epsilon: float = DEFAULT_BR_CRITIC_EPSILON,
        br_actor_epsilon: float = DEFAULT_BR_ACTOR_EPSILON,
        ma_window: int = DEFAULT_MA_WINDOW
    ):
        """
        Initialize convergence checker.

        All defaults are imported from src.trainers.config to avoid drift.

        Args:
            method: Training method ('lr', 'er', 'br')
            n_anneal: Steps required for annealing (gatekeeper)
            patience: Consecutive checks before stopping (default: DEFAULT_PATIENCE)
            lr_epsilon: Relative improvement threshold for LR (default: DEFAULT_LR_EPSILON)
            lr_window: Window size for LR improvement evaluation (default: DEFAULT_LR_WINDOW)
            er_epsilon: Absolute loss threshold for ER (unit-free, 1e-4 → ~1% accuracy)
            br_critic_epsilon: Critic loss threshold for BR (default: DEFAULT_BR_CRITIC_EPSILON)
            br_actor_epsilon: Actor relative improvement threshold for BR (default: DEFAULT_BR_ACTOR_EPSILON)
            ma_window: Moving average window size for smoothing (default: DEFAULT_MA_WINDOW)
        """
        self.method = method.lower()
        self.n_anneal = n_anneal
        self.patience = patience

        # Method-specific thresholds
        self.lr_epsilon = lr_epsilon
        self.lr_window = lr_window
        self.er_epsilon = er_epsilon
        self.br_critic_epsilon = br_critic_epsilon
        self.br_actor_epsilon = br_actor_epsilon
        self.ma_window = ma_window

        # State tracking
        self.state = StoppingState()
        # Buffer needs to hold enough for lookback (lr_window) plus averaging (ma_window)
        self._lr_loss_buffer = deque(maxlen=(lr_window + ma_window) * 2)

    def reset(self):
        """Reset convergence state."""
        self.state = StoppingState()
        self._lr_loss_buffer.clear()

    def check(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Check if training should stop.

        Args:
            step: Current training step
            metrics: Dictionary of validation metrics

        Returns:
            True if training should stop, False otherwise
        """
        # Step A: Gatekeeper for Annealing
        # If step < n_anneal, ignore all early stopping triggers
        if step < self.n_anneal:
            return False

        # Step B: Method-specific stopping rules
        criteria_met = self._check_method_criteria(metrics)

        # Update patience counter
        if criteria_met:
            self.state.patience_counter += 1
            logger.debug(f"Convergence criteria met. Patience: {self.state.patience_counter}/{self.patience}")
        else:
            self.state.patience_counter = 0

        # Stop if patience exhausted
        should_stop = self.state.patience_counter >= self.patience

        if should_stop:
            logger.info(f"Early stopping triggered at step {step}. Method: {self.method}")

        return should_stop

    def _check_method_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        Check method-specific convergence criteria.

        """
        if self.method == 'lr':
            return self._check_lr_criteria(metrics)
        elif self.method == 'er':
            return self._check_er_criteria(metrics)
        elif self.method == 'br':
            return self._check_br_criteria(metrics)
        else:
            logger.warning(f"Unknown method '{self.method}', no stopping criteria")
            return False

    def _check_lr_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        LR Method: Relative Improvement Plateau.

        Criteria: Stop if relative improvement is negligible:
            (L_LR(j) - L_LR(j-s)) / |L_LR(j-s)| < epsilon_LR

        """
        loss_key = 'loss_LR'
        if loss_key not in metrics:
            return False

        current_loss = metrics[loss_key]
        self._lr_loss_buffer.append(current_loss)

        # Need enough samples for current MA and past MA
        # We need samples at indices: [current], ..., [current - ma_window + 1]
        # And: [current - lr_window], ..., [current - lr_window - ma_window + 1]
        # Total history required: lr_window + ma_window
        if len(self._lr_loss_buffer) < self.lr_window + self.ma_window:
            return False

        # Helper to compute MA ending at specific index (0-based from end of deque)
        def get_ma(buffer, end_idx, width):
            # end_idx is 0 for last element, 1 for second to last, etc.
            # python slicing: buffer[len-1-end_idx : len-1-end_idx-width : -1] is hard with deque
            # cast to list for slicing validity (deque slicing is limited)
            data = list(buffer)
            start = len(data) - 1 - end_idx
            stop = start - width
            if stop < -1:
                segment = data[start::-1][:width] # Take up to width elements backwards
            else:
                segment = []
                for i in range(width):
                    segment.append(data[start - i])
            return sum(segment) / len(segment)

        # Current MA: average of last ma_window elements
        current_ma = get_ma(self._lr_loss_buffer, 0, self.ma_window)

        # Past MA: average of ma_window elements ending at lr_window steps ago
        past_ma = get_ma(self._lr_loss_buffer, self.lr_window, self.ma_window)

        # Relative improvement (use DEFAULT_DIVISION_EPSILON to avoid division by zero)
        if abs(past_ma) < DEFAULT_DIVISION_EPSILON:
            return False

        relative_improvement = (current_ma - past_ma) / abs(past_ma)

        # For LR, we minimize negative reward, so improvement is when loss decreases
        # A small relative change indicates plateau
        return abs(relative_improvement) < self.lr_epsilon

    def _check_er_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        ER Method: Zero-Tolerance Plateau.

        Criteria: L_ER < epsilon_ER

        For unit-free residuals f = 1 - β*m/χ:
        - L_ER = E[f²] where f is the fractional Euler deviation
        - epsilon_ER = 1e-4 corresponds to ~1% average Euler accuracy
        - epsilon_ER = 1e-5 corresponds to ~0.3% average Euler accuracy
        - epsilon_ER = 1e-3 corresponds to ~3% average Euler accuracy

        Reference: report_brief.md lines 599-604, 723-784
        """
        loss_key = 'loss_ER'
        if loss_key not in metrics:
            return False

        current_loss = metrics[loss_key]

        # ER loss should be close to zero (root-finding)
        return abs(current_loss) < self.er_epsilon

    def _check_br_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        BR Method: Dual-Condition Convergence.

        Criteria: Both conditions must be met:
            1. Critic Accuracy: Bellman Residual < epsilon_crit
            2. Policy Stability: Actor value relative improvement < epsilon_act

        """
        critic_key = 'loss_critic'
        actor_key = 'loss_actor'

        if critic_key not in metrics or actor_key not in metrics:
            return False

        critic_loss = metrics[critic_key]
        actor_loss = metrics[actor_key]

        # Condition 1: Critic accuracy (loss close to zero)
        critic_converged = abs(critic_loss) < self.br_critic_epsilon

        # Condition 2: Actor stability (use similar logic to LR)
        # Track actor loss history
        self.state.loss_history.append(actor_loss)

        actor_converged = False
        if len(self.state.loss_history) >= self.lr_window + 1:
            past_actor = self.state.loss_history[-self.lr_window - 1]
            if abs(past_actor) > DEFAULT_DIVISION_EPSILON:
                relative_change = (actor_loss - past_actor) / abs(past_actor)
                actor_converged = abs(relative_change) < self.br_actor_epsilon

        # Both conditions must be met
        return critic_converged and actor_converged


def create_convergence_checker(
    method_name: str,
    n_anneal: int,
    early_stopping_config
) -> Optional[ConvergenceChecker]:
    """
    Factory function to create a ConvergenceChecker from config.

    Args:
        method_name: Full method name (e.g., 'basic_lr', 'risky_br')
        n_anneal: Annealing steps from AnnealingSchedule
        early_stopping_config: EarlyStoppingConfig object

    Returns:
        ConvergenceChecker if early stopping is enabled, None otherwise
    """
    if early_stopping_config is None or not early_stopping_config.enabled:
        return None

    # Extract method family from name
    method_parts = method_name.lower().split('_')
    method_family = None
    for part in ['lr', 'er', 'br']:
        if part in method_parts:
            method_family = part
            break

    if method_family is None:
        logger.warning(f"Cannot determine method family from '{method_name}', no convergence checker created")
        return None

    return ConvergenceChecker(
        method=method_family,
        n_anneal=n_anneal,
        patience=early_stopping_config.patience,
        lr_epsilon=early_stopping_config.lr_epsilon,
        lr_window=early_stopping_config.lr_window,
        er_epsilon=early_stopping_config.er_epsilon,
        br_critic_epsilon=early_stopping_config.br_critic_epsilon,
        br_actor_epsilon=early_stopping_config.br_actor_epsilon,
        ma_window=early_stopping_config.ma_window
    )
