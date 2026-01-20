"""
src/dnn/evaluation/

Modular evaluation package for DNN models.
Refactored from monolithic src/dnn/evaluation.py.
"""

# Re-export efficient submodules
from src.dnn.evaluation.common import (
    get_eval_grids,
    compute_moments,
    compare_moments,
    find_steady_state_k,
)

from src.dnn.evaluation.wrappers import (
    evaluate_basic_policy,
    evaluate_basic_value,
    evaluate_risky_policy,
    evaluate_risky_value,
)

from src.dnn.evaluation.simulation import (
    simulate_policy_path,
    evaluate_policy_return,
)

from src.dnn.evaluation.residuals import (
    eval_euler_residual_basic,
)
