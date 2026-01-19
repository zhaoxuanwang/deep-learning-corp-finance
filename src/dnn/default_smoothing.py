"""
src/dnn/default_smoothing.py

Default smoothing for Risky Debt model pricing.
Implements epsilon_D schedule and smooth default probability.

The smooth default probability p^D is used ONLY inside the pricing loss,
not in the actual value function (which uses hard limited liability).

Reference: outline_v2.md lines 122-135, 317-318
"""

import tensorflow as tf


class DefaultSmoothingSchedule:
    """
    Manages epsilon_D temperature schedule for soft default probability.
    
    The smooth default probability is:
        p^D = sigmoid(-V_tilde / epsilon_D)
    
    As epsilon_D -> 0, p^D -> indicator{V_tilde < 0}.
    
    Schedule:
    - Initialize: epsilon_D = epsilon_D_0
    - Update once per outer iteration: epsilon_D = max(epsilon_D_min, d * epsilon_D)
    
    Numerical stability:
    - Clip logit u = -V_tilde / epsilon_D to [-u_max, u_max] before sigmoid
    
    Args:
        epsilon_D_0: Initial temperature (default 0.1)
        epsilon_D_min: Minimum temperature (default 1e-4)
        decay_d: Decay factor per outer iteration (default 0.99)
        u_max: Logit clipping bound (default 10.0)
    
    Reference: outline_v2.md lines 122-135
    """
    
    def __init__(
        self,
        epsilon_D_0: float = 0.1,
        epsilon_D_min: float = 1e-4,
        decay_d: float = 0.99,
        u_max: float = 10.0
    ):
        self.epsilon_D_0 = epsilon_D_0
        self.epsilon_D_min = epsilon_D_min
        self.decay_d = decay_d
        self.u_max = u_max
        
        # Current epsilon_D (mutable state)
        self._epsilon_D = epsilon_D_0
    
    @property
    def epsilon_D(self) -> float:
        """Current epsilon_D value."""
        return self._epsilon_D
    
    def reset(self):
        """Reset epsilon_D to initial value."""
        self._epsilon_D = self.epsilon_D_0
    
    def update(self):
        """
        Update epsilon_D. Call once per outer iteration.
        
        epsilon_D = max(epsilon_D_min, d * epsilon_D)
        
        Reference: outline_v2.md line 128
        """
        self._epsilon_D = max(self.epsilon_D_min, self.decay_d * self._epsilon_D)
    
    def compute_default_prob(self, V_tilde: tf.Tensor) -> tf.Tensor:
        """
        Compute smooth default probability.
        
        p^D = sigmoid(clip(-V_tilde / epsilon_D, -u_max, u_max))
        
        Args:
            V_tilde: Latent value (can be positive or negative)
        
        Returns:
            p_default: Smooth default probability in [0, 1]
        
        Reference: outline_v2.md lines 124, 134
        """
        # u = -V_tilde / epsilon_D
        u = -V_tilde / (self._epsilon_D + 1e-10)
        
        # Clip for numerical stability
        u_clipped = tf.clip_by_value(u, -self.u_max, self.u_max)
        
        # p^D = sigmoid(u)
        p_default = tf.nn.sigmoid(u_clipped)
        
        return p_default
    
    def get_state(self) -> dict:
        """Get current schedule state for checkpointing."""
        return {"epsilon_D": self._epsilon_D}
    
    def set_state(self, state: dict):
        """Restore schedule state from checkpoint."""
        self._epsilon_D = state["epsilon_D"]


def compute_smooth_default_prob(
    V_tilde: tf.Tensor,
    epsilon_D: float,
    u_max: float = 10.0
) -> tf.Tensor:
    """
    Functional version of smooth default probability computation.
    
    p^D = sigmoid(clip(-V_tilde / epsilon_D, -u_max, u_max))
    
    Args:
        V_tilde: Latent value
        epsilon_D: Current temperature
        u_max: Logit clipping bound
    
    Returns:
        p_default: Smooth default probability
    """
    u = -V_tilde / (epsilon_D + 1e-10)
    u_clipped = tf.clip_by_value(u, -u_max, u_max)
    return tf.nn.sigmoid(u_clipped)
