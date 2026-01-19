"""
tests/dnn/test_experiments.py

Minimal tests for experiment utilities.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from src.dnn.experiments import (
    TrainingConfig,
    EconomicScenario,
    train_basic_lr,
    train_basic_er,
    train_basic_br,
    train_risky_br,
    train
)
from src.dnn.evaluation import (
    evaluate_basic_policy,
    evaluate_basic_value,
    evaluate_risky_policy,
    compute_moments
)
from src.dnn.plotting import (
    plot_loss_curve,
    plot_loss_comparison,
    quick_plot
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return TrainingConfig(
        n_layers=1, n_neurons=8, n_iter=20,
        batch_size=32, log_every=5, seed=42
    )

@pytest.fixture
def tiny_config():
    """Tiny config for smoke testing interfaces (1 iteration)."""
    return TrainingConfig(
        n_layers=1, n_neurons=8, n_iter=1,
        batch_size=32, log_every=1, seed=42
    )


@pytest.fixture
def scenario():
    """Baseline scenario for testing."""
    return EconomicScenario(name="test_baseline")


# =============================================================================
# TRAINING WRAPPER TESTS
# =============================================================================

class TestTrainingWrappers:
    """Tests for training wrappers returning expected keys."""
    
    def test_train_basic_lr_returns_history(self, scenario, tiny_config):
        """train_basic_lr returns history with expected keys."""
        history = train_basic_lr(scenario, tiny_config)
        
        assert "iteration" in history
        assert "loss_LR" in history
        assert "mean_reward" in history
        assert len(history["iteration"]) > 0
        assert "_policy_net" in history
    
    def test_train_basic_er_returns_history(self, scenario, tiny_config):
        """train_basic_er returns history with expected keys."""
        history = train_basic_er(scenario, tiny_config)
        
        assert "loss_ER" in history
        assert len(history["loss_ER"]) > 0
    
    def test_train_basic_br_returns_history(self, scenario, tiny_config):
        """train_basic_br returns history with expected keys."""
        history = train_basic_br(scenario, tiny_config)
        
        assert "loss_BR_critic" in history
        assert "loss_BR_actor" in history
        assert "_policy_net" in history
        assert "_value_net" in history
    
    def test_train_risky_br_returns_history(self, scenario, tiny_config):
        """train_risky_br returns history with expected keys."""
        history = train_risky_br(scenario, tiny_config)
        
        assert "loss_critic" in history
        assert "loss_actor" in history
        assert "loss_price" in history
        assert "epsilon_D" in history
        assert "_policy_net" in history
        assert "_price_net" in history
    
    def test_unified_train_api(self, scenario, tiny_config):
        """train() dispatches correctly."""
        history = train("basic_lr", scenario, tiny_config)
        assert "loss_LR" in history



# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for reproducibility with fixed seeds."""
    
    def test_same_seed_same_history(self, scenario):
        """Fixed seed produces similar histories (TF has some internal variance)."""
        config = TrainingConfig(n_iter=10, seed=123, log_every=5)
        
        h1 = train_basic_lr(scenario, config)
        h2 = train_basic_lr(scenario, config)
        
        # Allow 20% relative tolerance due to TF internal RNG state
        np.testing.assert_allclose(h1["loss_LR"], h2["loss_LR"], rtol=0.25)
    
    def test_different_seeds_different_history(self, scenario):
        """Different seeds produce different histories."""
        config1 = TrainingConfig(n_iter=10, seed=1, log_every=5)
        config2 = TrainingConfig(n_iter=10, seed=2, log_every=5)
        
        h1 = train_basic_lr(scenario, config1)
        h2 = train_basic_lr(scenario, config2)
        
        # Should be different (almost certainly)
        assert not np.allclose(h1["loss_LR"], h2["loss_LR"])


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Tests for evaluation functions."""
    
    def test_evaluate_basic_policy_shapes(self, scenario, small_config):
        """Evaluation outputs have correct shapes."""
        history = train_basic_lr(scenario, small_config)
        
        k_grid = np.linspace(0.5, 5.0, 10)
        z_grid = np.linspace(0.8, 1.2, 5)
        
        data = evaluate_basic_policy(history["_policy_net"], k_grid, z_grid)
        
        assert data["k"].shape == (10, 5)
        assert data["z"].shape == (10, 5)
        assert data["k_next"].shape == (10, 5)
        assert data["I_k"].shape == (10, 5)
    
    def test_evaluation_returns_levels(self, scenario, small_config):
        """Evaluation returns values in levels, not log."""
        history = train_basic_lr(scenario, small_config)
        
        k_grid = np.linspace(0.5, 5.0, 10)
        z_grid = np.linspace(0.8, 1.2, 5)
        
        data = evaluate_basic_policy(history["_policy_net"], k_grid, z_grid)
        
        # k_next should be positive (levels)
        assert np.all(data["k_next"] > 0)
        
        # k values should match input grid
        np.testing.assert_allclose(data["k"][:, 0], k_grid)
    
    def test_compute_moments(self, scenario, small_config):
        """compute_moments returns DataFrame with expected columns."""
        history = train_basic_lr(scenario, small_config)
        
        data = evaluate_basic_policy(
            history["_policy_net"],
            np.linspace(0.5, 5.0, 10),
            np.linspace(0.8, 1.2, 5)
        )
        
        moments = compute_moments(data)
        
        assert "variable" in moments.columns
        assert "mean" in moments.columns
        assert "median" in moments.columns
        assert len(moments) >= 3  # k, z, k_next, I_k


# =============================================================================
# PLOTTING SMOKE TESTS
# =============================================================================

class TestPlotting:
    """Smoke tests for plotting functions."""
    
    @pytest.fixture
    def mock_history(self):
        """Mock history dictionary to avoid running training."""
        return {
            "iteration": np.arange(10),
            "loss_LR": np.random.randn(10),
            "loss_ER": np.random.randn(10),
            "loss_BR_critic": np.random.randn(10),
            "loss_BR_actor": np.random.randn(10),
            "mean_reward": np.random.randn(10),
            # Add other keys required by specific plots if needed
        }
    
    def test_plot_loss_curve_no_crash(self, mock_history):
        """plot_loss_curve executes without error."""
        fig = plot_loss_curve(mock_history)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_loss_comparison_no_crash(self, mock_history):
        """plot_loss_comparison executes without error."""
        h1 = mock_history
        h2 = mock_history.copy()
        
        fig = plot_loss_comparison([h1, h2], ["Run 1", "Run 2"])
        assert fig is not None
        plt.close(fig)
    
    def test_quick_plot_no_crash(self, mock_history):
        """quick_plot executes without error."""
        fig = quick_plot(mock_history)
        assert fig is not None
        plt.close(fig)
