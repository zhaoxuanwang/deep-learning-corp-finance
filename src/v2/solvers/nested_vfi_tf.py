"""TF-native nested VFI for the risky-debt model."""

from __future__ import annotations

import contextlib
import functools
import time
from typing import Dict

import numpy as np
import tensorflow as tf

from src.v2.solvers.config import NestedVFIConfig, NestedVFITFRuntimeConfig
from src.v2.solvers.grid import tauchen_transition_matrix_tf
from src.v2.solvers.nested_vfi_np import _build_nested_vfi_grids, _AUTARKY_RATE


def _tf_dtype_from_name(name: str) -> tf.dtypes.DType:
    if name == "float32":
        return tf.float32
    if name == "float64":
        return tf.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_device_name(device: str | None) -> str:
    if not device:
        return "CPU:0"
    if "device:" in device:
        return device.rsplit("device:", 1)[-1]
    return device


def _pricing_discount_sup_norm_diff_tf(
    r_new: tf.Tensor,
    r_old: tf.Tensor,
) -> float:
    eps = tf.cast(1e-12, r_new.dtype)
    discount_new = tf.where(
        tf.math.is_finite(r_new),
        1.0 / tf.maximum(1.0 + r_new, eps),
        tf.zeros_like(r_new),
    )
    discount_old = tf.where(
        tf.math.is_finite(r_old),
        1.0 / tf.maximum(1.0 + r_old, eps),
        tf.zeros_like(r_old),
    )
    return float(tf.reduce_max(tf.abs(discount_new - discount_old)).numpy())


@functools.lru_cache(maxsize=None)
def _get_inner_bellman_solver_tf(
    n_choice: int,
    choice_block_size: int,
    jit_compile: bool,
    record_inner_history: bool,
):
    """Return a cached compiled inner Bellman solver for fixed choice shape."""

    if record_inner_history:

        @tf.function(jit_compile=jit_compile, reduce_retracing=True)
        def _solve(
            prob_matrix: tf.Tensor,
            value_init: tf.Tensor,
            state_base: tf.Tensor,
            state_choice_outlay: tf.Tensor,
            choice_financing_value: tf.Tensor,
            beta: tf.Tensor,
            tol_inner: tf.Tensor,
            max_iter_inner: tf.Tensor,
            cost_inject_fixed: tf.Tensor,
            cost_inject_linear: tf.Tensor,
        ):
            inner0 = tf.constant(0, dtype=tf.int32)
            policy0 = tf.zeros(tf.shape(value_init), dtype=tf.int32)
            diff0 = tf.cast(np.inf, value_init.dtype)
            converged0 = tf.constant(False)
            history0 = tf.TensorArray(
                dtype=value_init.dtype,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
            )

            def cond(inner, value, policy_idx, diff, converged, history):
                del value, policy_idx, diff, history
                return tf.logical_and(inner < max_iter_inner, tf.logical_not(converged))

            def body(inner, value, policy_idx, diff, converged, history):
                del policy_idx, diff, converged
                expected_continuation = tf.linalg.matmul(prob_matrix, value)
                best_rhs = tf.fill(tf.shape(value), tf.cast(-1e30, value.dtype))
                best_idx = tf.zeros(tf.shape(value), dtype=tf.int32)

                for block_start in range(0, n_choice, choice_block_size):
                    block_end = min(block_start + choice_block_size, n_choice)
                    cash_flow_block = (
                        state_base[:, :, None]
                        - state_choice_outlay[None, :, block_start:block_end]
                        + choice_financing_value[:, None, block_start:block_end]
                    )
                    shortfall = tf.nn.relu(-cash_flow_block)
                    reward_block = (
                        cash_flow_block
                        - cost_inject_linear * shortfall
                        - cost_inject_fixed
                        * tf.cast(shortfall > 0.0, value.dtype)
                    )
                    rhs_block = (
                        reward_block
                        + beta * expected_continuation[:, None, block_start:block_end]
                    )
                    rhs_block_max = tf.reduce_max(rhs_block, axis=2)
                    rhs_block_idx = (
                        tf.argmax(rhs_block, axis=2, output_type=tf.int32)
                        + block_start
                    )
                    better = rhs_block_max > best_rhs
                    best_rhs = tf.where(better, rhs_block_max, best_rhs)
                    best_idx = tf.where(better, rhs_block_idx, best_idx)

                value_next = tf.nn.relu(best_rhs)
                diff_next = tf.reduce_max(tf.abs(value_next - value))
                converged_next = diff_next < tol_inner
                history = history.write(inner, diff_next)
                return inner + 1, value_next, best_idx, diff_next, converged_next, history

            (
                n_inner,
                value,
                policy_idx,
                inner_diff,
                converged,
                history,
            ) = tf.while_loop(
                cond,
                body,
                [inner0, value_init, policy0, diff0, converged0, history0],
                parallel_iterations=1,
            )
            return value, policy_idx, converged, n_inner, inner_diff, history.stack()

    else:

        @tf.function(jit_compile=jit_compile, reduce_retracing=True)
        def _solve(
            prob_matrix: tf.Tensor,
            value_init: tf.Tensor,
            state_base: tf.Tensor,
            state_choice_outlay: tf.Tensor,
            choice_financing_value: tf.Tensor,
            beta: tf.Tensor,
            tol_inner: tf.Tensor,
            max_iter_inner: tf.Tensor,
            cost_inject_fixed: tf.Tensor,
            cost_inject_linear: tf.Tensor,
        ):
            inner0 = tf.constant(0, dtype=tf.int32)
            policy0 = tf.zeros(tf.shape(value_init), dtype=tf.int32)
            diff0 = tf.cast(np.inf, value_init.dtype)
            converged0 = tf.constant(False)

            def cond(inner, value, policy_idx, diff, converged):
                del value, policy_idx, diff
                return tf.logical_and(inner < max_iter_inner, tf.logical_not(converged))

            def body(inner, value, policy_idx, diff, converged):
                del policy_idx, diff, converged
                expected_continuation = tf.linalg.matmul(prob_matrix, value)
                best_rhs = tf.fill(tf.shape(value), tf.cast(-1e30, value.dtype))
                best_idx = tf.zeros(tf.shape(value), dtype=tf.int32)

                for block_start in range(0, n_choice, choice_block_size):
                    block_end = min(block_start + choice_block_size, n_choice)
                    cash_flow_block = (
                        state_base[:, :, None]
                        - state_choice_outlay[None, :, block_start:block_end]
                        + choice_financing_value[:, None, block_start:block_end]
                    )
                    shortfall = tf.nn.relu(-cash_flow_block)
                    reward_block = (
                        cash_flow_block
                        - cost_inject_linear * shortfall
                        - cost_inject_fixed
                        * tf.cast(shortfall > 0.0, value.dtype)
                    )
                    rhs_block = (
                        reward_block
                        + beta * expected_continuation[:, None, block_start:block_end]
                    )
                    rhs_block_max = tf.reduce_max(rhs_block, axis=2)
                    rhs_block_idx = (
                        tf.argmax(rhs_block, axis=2, output_type=tf.int32)
                        + block_start
                    )
                    better = rhs_block_max > best_rhs
                    best_rhs = tf.where(better, rhs_block_max, best_rhs)
                    best_idx = tf.where(better, rhs_block_idx, best_idx)

                value_next = tf.nn.relu(best_rhs)
                diff_next = tf.reduce_max(tf.abs(value_next - value))
                converged_next = diff_next < tol_inner
                return inner + 1, value_next, best_idx, diff_next, converged_next

            (
                n_inner,
                value,
                policy_idx,
                inner_diff,
                converged,
            ) = tf.while_loop(
                cond,
                body,
                [inner0, value_init, policy0, diff0, converged0],
                parallel_iterations=1,
            )
            empty_history = tf.zeros([0], dtype=value.dtype)
            return value, policy_idx, converged, n_inner, inner_diff, empty_history

    return _solve


def _compute_pricing_update_tf(
    env,
    prob_matrix: tf.Tensor,
    value_3d: tf.Tensor,
    recovery_flat: tf.Tensor,
    choice_b: tf.Tensor,
    n_z: int,
    n_k: int,
    n_b: int,
    dtype: tf.dtypes.DType,
):
    eps = tf.cast(1e-12, dtype)
    r_rf = tf.cast(env.econ.interest_rate, dtype)
    default_mask = value_3d <= 0.0
    default_flat = tf.reshape(tf.cast(default_mask, dtype), [n_z, -1])
    solvent_flat = 1.0 - default_flat

    expected_recovery = tf.linalg.matmul(prob_matrix, default_flat * recovery_flat)
    solvent_probability = tf.linalg.matmul(prob_matrix, solvent_flat)

    b_flat = choice_b[None, :]
    safe_b = tf.maximum(b_flat, eps)
    safe_solvent = tf.maximum(solvent_probability, eps)
    b_positive = b_flat > eps
    funded = tf.logical_and(b_positive, solvent_probability > eps)

    funded_values = (
        ((1.0 + r_rf) - expected_recovery / safe_b) / safe_solvent
    ) - 1.0
    r_tilde_new = tf.fill(tf.shape(expected_recovery), tf.cast(np.inf, dtype))
    r_tilde_new = tf.where(
        tf.broadcast_to(tf.logical_not(b_positive), tf.shape(expected_recovery)),
        tf.fill(tf.shape(expected_recovery), r_rf),
        r_tilde_new,
    )
    r_tilde_new = tf.where(funded, funded_values, r_tilde_new)

    return (
        tf.reshape(r_tilde_new, [n_z, n_k, n_b]),
        default_mask,
        tf.reshape(expected_recovery, [n_z, n_k, n_b]),
        tf.reshape(solvent_probability, [n_z, n_k, n_b]),
    )


def _zero_profit_residual_tf(
    env,
    b_grid: tf.Tensor,
    r_tilde_grid: tf.Tensor,
    expected_recovery: tf.Tensor,
    solvent_probability: tf.Tensor,
    dtype: tf.dtypes.DType,
):
    eps = tf.cast(1e-12, dtype)
    b_3d = tf.broadcast_to(
        b_grid[None, None, :],
        tf.shape(r_tilde_grid),
    )
    funded_mask = tf.logical_and(
        b_3d > eps,
        tf.logical_and(
            tf.math.is_finite(r_tilde_grid),
            solvent_probability > eps,
        ),
    )
    lhs = b_3d * tf.cast(1.0 + env.econ.interest_rate, dtype)
    rhs_values = (
        expected_recovery
        + b_3d * (1.0 + r_tilde_grid) * solvent_probability
    )
    nan_tensor = tf.fill(tf.shape(r_tilde_grid), tf.cast(np.nan, dtype))
    residual = tf.where(funded_mask, lhs - rhs_values, nan_tensor)
    return residual, funded_mask


def _extract_policy_tf(
    env,
    choice_product: tf.Tensor,
    endo_product: tf.Tensor,
    policy_idx: tf.Tensor,
    dtype: tf.dtypes.DType,
):
    policy_endo = tf.gather(choice_product, policy_idx)
    current_k = endo_product[:, 0][None, :]
    investment = policy_endo[..., 0] - tf.cast(
        1.0 - env.econ.depreciation_rate,
        dtype,
    ) * current_k
    policy_action = tf.stack([investment, policy_endo[..., 1]], axis=-1)
    return policy_action, policy_endo


def solve_nested_vfi_tf(
    env,
    train_dataset: Dict[str, tf.Tensor] | None = None,
    config: NestedVFIConfig | None = None,
    eval_callback=None,
    runtime_config: NestedVFITFRuntimeConfig | None = None,
    value_init: np.ndarray | None = None,
    r_tilde_init: np.ndarray | None = None,
) -> Dict:
    """Solve the risky-debt benchmark via a TF-native blocked nested VFI.

    Args:
        value_init:   Optional initial value function, shape (n_z, n_state).
                      When None (default), starts from zeros (cold start).
                      Pass the converged value from a previous solve at
                      nearby parameters to reduce iterations.
        r_tilde_init: Optional initial risky-rate schedule, shape
                      (n_z, n_k, n_b).  When None (default), uses pessimistic
                      autarky initialization.  Pass the converged pricing
                      from a previous solve for warm-starting.
    """

    del train_dataset
    config = config or NestedVFIConfig()
    runtime_config = runtime_config or NestedVFITFRuntimeConfig()
    dtype = _tf_dtype_from_name(runtime_config.dtype)

    grids = _build_nested_vfi_grids(env, config.grid)
    n_z = len(grids["exo_grids_1d"][0])
    n_k = len(grids["endo_grids_1d"][0])
    n_b = len(grids["endo_grids_1d"][1])
    n_state = int(grids["endo_product"].shape[0])
    n_choice = int(grids["choice_product"].shape[0])

    device_ctx = (
        tf.device(runtime_config.device)
        if runtime_config.device is not None
        else contextlib.nullcontext()
    )

    with device_ctx:
        prob_matrix = tauchen_transition_matrix_tf(
            env.shocks,
            grids["exo_grids_1d"],
            dtype=dtype,
        )
        z_grid = tf.convert_to_tensor(grids["exo_grids_1d"][0], dtype=dtype)
        k_grid = tf.convert_to_tensor(grids["endo_grids_1d"][0], dtype=dtype)
        b_grid = tf.convert_to_tensor(grids["endo_grids_1d"][1], dtype=dtype)
        endo_product = tf.convert_to_tensor(grids["endo_product"], dtype=dtype)
        choice_product = tf.convert_to_tensor(grids["choice_product"], dtype=dtype)

        current_k = endo_product[:, 0]
        current_b = endo_product[:, 1]
        choice_k = choice_product[:, 0]
        choice_b = choice_product[:, 1]

        eps = tf.cast(1e-12, dtype)
        delta = tf.cast(env.econ.depreciation_rate, dtype)
        tax = tf.cast(env.econ.tax, dtype)
        alpha = tf.cast(env.econ.production_elasticity, dtype)
        beta = tf.cast(env.discount(), dtype)
        rf = tf.cast(env.econ.interest_rate, dtype)
        fixed_cost = tf.cast(env.econ.cost_inject_fixed, dtype)
        linear_cost = tf.cast(env.econ.cost_inject_linear, dtype)
        tol_inner = tf.cast(config.tol_inner, dtype)
        max_iter_inner = tf.constant(config.max_iter_inner, dtype=tf.int32)

        investment_state_choice = (
            choice_k[None, :] - (1.0 - delta) * current_k[:, None]
        )
        adjustment_cost = (
            0.5
            * tf.cast(env.econ.cost_convex, dtype)
            * tf.square(investment_state_choice)
            / tf.maximum(current_k[:, None], eps)
        )
        state_choice_outlay = investment_state_choice + adjustment_cost
        after_tax_profit = (
            (1.0 - tax) * z_grid[:, None] * tf.pow(current_k[None, :], alpha)
        )
        state_base = after_tax_profit - current_b[None, :]

        recovery = tf.cast(
            env.recovery_value(k_grid[None, :, None], z_grid[:, None, None]),
            dtype,
        )
        recovery_flat = tf.reshape(tf.repeat(recovery, repeats=n_b, axis=2), [n_z, -1])

        if value_init is not None:
            value = tf.cast(
                tf.reshape(value_init, [n_z, n_state]), dtype
            )
        else:
            value = tf.zeros([n_z, n_state], dtype=dtype)
        policy_idx = tf.zeros([n_z, n_state], dtype=tf.int32)

        if r_tilde_init is not None:
            r_tilde_grid = tf.cast(
                tf.reshape(r_tilde_init, [n_z, n_k, n_b]), dtype
            )
        else:
            # Pessimistic (autarky) initialization, matching the NumPy
            # version: lenders refuse to lend at reasonable rates for
            # positive debt, seeding a non-empty default region.
            b_grid_np = np.asarray(grids["endo_grids_1d"][1], dtype=np.float64)
            r_init_np = np.full(
                (n_z, n_k, n_b), env.econ.interest_rate, dtype=np.float64
            )
            r_init_np[:, :, b_grid_np > 1e-12] = _AUTARKY_RATE
            r_tilde_grid = tf.cast(r_init_np, dtype)

        inner_solver = _get_inner_bellman_solver_tf(
            n_choice=n_choice,
            choice_block_size=runtime_config.choice_block_size,
            jit_compile=runtime_config.jit_compile,
            record_inner_history=runtime_config.record_inner_history,
        )

        start_time = time.perf_counter()
        inner_diff_history: list[float] = []
        outer_value_diff_history: list[float] = []
        pricing_diff_history: list[float] = []
        eval_history = {}
        converged_inner = False
        converged_outer = False
        n_inner_last = 0
        stop_reason = "max_outer"
        default_mask = tf.zeros([n_z, n_k, n_b], dtype=tf.bool)
        expected_recovery = tf.zeros([n_z, n_k, n_b], dtype=dtype)
        solvent_probability = tf.zeros([n_z, n_k, n_b], dtype=dtype)
        n_outer = 0
        inner_diff = float("inf")

        debt_discount = tf.where(
            tf.math.is_finite(r_tilde_grid),
            1.0 / tf.maximum(1.0 + r_tilde_grid, eps),
            tf.zeros_like(r_tilde_grid),
        )
        debt_discount_flat = tf.reshape(debt_discount, [n_z, n_choice])
        choice_financing_value = (
            choice_b[None, :] * debt_discount_flat
            + tax * choice_b[None, :] * (1.0 - debt_discount_flat) / (1.0 + rf)
        )

        (
            value,
            policy_idx,
            converged_inner_tf,
            n_inner_last_tf,
            inner_diff_tf,
            inner_history_tensor,
        ) = inner_solver(
            prob_matrix,
            value,
            state_base,
            state_choice_outlay,
            choice_financing_value,
            beta,
            tol_inner,
            max_iter_inner,
            fixed_cost,
            linear_cost,
        )
        converged_inner = bool(converged_inner_tf.numpy())
        n_inner_last = int(n_inner_last_tf.numpy())
        inner_diff = float(inner_diff_tf.numpy())
        if runtime_config.record_inner_history:
            inner_diff_history.extend(
                float(x) for x in inner_history_tensor.numpy().tolist()
            )

        if not converged_inner:
            stop_reason = "max_inner"
        else:
            value_prev_outer = tf.identity(value)

            for outer in range(config.max_iter_outer):
                value_prev_3d = tf.reshape(value_prev_outer, [n_z, n_k, n_b])
                (
                    r_tilde_new,
                    default_mask,
                    expected_recovery,
                    solvent_probability,
                ) = _compute_pricing_update_tf(
                    env,
                    prob_matrix,
                    value_prev_3d,
                    recovery_flat,
                    choice_b,
                    n_z,
                    n_k,
                    n_b,
                    dtype,
                )
                pricing_diff = _pricing_discount_sup_norm_diff_tf(
                    r_tilde_new,
                    r_tilde_grid,
                )

                debt_discount = tf.where(
                    tf.math.is_finite(r_tilde_new),
                    1.0 / tf.maximum(1.0 + r_tilde_new, eps),
                    tf.zeros_like(r_tilde_new),
                )
                debt_discount_flat = tf.reshape(debt_discount, [n_z, n_choice])
                choice_financing_value = (
                    choice_b[None, :] * debt_discount_flat
                    + tax * choice_b[None, :] * (1.0 - debt_discount_flat) / (1.0 + rf)
                )

                (
                    value,
                    policy_idx,
                    converged_inner_tf,
                    n_inner_last_tf,
                    inner_diff_tf,
                    inner_history_tensor,
                ) = inner_solver(
                    prob_matrix,
                    value_prev_outer,
                    state_base,
                    state_choice_outlay,
                    choice_financing_value,
                    beta,
                    tol_inner,
                    max_iter_inner,
                    fixed_cost,
                    linear_cost,
                )
                converged_inner = bool(converged_inner_tf.numpy())
                n_inner_last = int(n_inner_last_tf.numpy())
                inner_diff = float(inner_diff_tf.numpy())
                if runtime_config.record_inner_history:
                    inner_diff_history.extend(
                        float(x) for x in inner_history_tensor.numpy().tolist()
                    )

                outer_value_diff = float(
                    tf.reduce_max(tf.abs(value - value_prev_outer)).numpy()
                )

                outer_value_diff_history.append(outer_value_diff)
                pricing_diff_history.append(pricing_diff)
                n_outer = outer + 1
                r_tilde_grid = r_tilde_new

                value_3d = tf.reshape(value, [n_z, n_k, n_b])
                default_mask = value_3d <= 0.0
                policy_action, policy_endo = _extract_policy_tf(
                    env,
                    choice_product,
                    endo_product,
                    policy_idx,
                    dtype,
                )
                partial_result = {
                    "value": value,
                    "policy_action": policy_action,
                    "policy_endo": policy_endo,
                    "grids": grids,
                    "r_tilde_grid": r_tilde_grid,
                    "default_mask": default_mask,
                    "backend": "tensorflow",
                    "dtype": runtime_config.dtype,
                    "device": _normalize_device_name(value.device),
                }

                if eval_callback is not None:
                    metrics = eval_callback(outer, partial_result) or {}
                    elapsed = time.perf_counter() - start_time
                    eval_history.setdefault("outer_iter", []).append(outer)
                    eval_history.setdefault("elapsed_sec", []).append(elapsed)
                    eval_history.setdefault("n_inner", []).append(n_inner_last)
                    eval_history.setdefault("inner_diff", []).append(inner_diff)
                    eval_history.setdefault("outer_value_diff", []).append(outer_value_diff)
                    eval_history.setdefault("pricing_diff", []).append(pricing_diff)
                    for key, val in metrics.items():
                        eval_history.setdefault(key, []).append(val)

                inner_status = (
                    "converged"
                    if converged_inner
                    else f"max({config.max_iter_inner})"
                )
                print(
                    f"Nested VFI TF outer={outer:3d} | "
                    f"inner={inner_status} ({n_inner_last} iters, diff={inner_diff:.2e}) | "
                    f"outer_value_diff={outer_value_diff:.2f} | "
                    f"pricing_diff={pricing_diff:.2f}"
                )

                if not converged_inner:
                    stop_reason = "max_inner"
                    break

                if outer_value_diff < config.tol_outer_value:
                    converged_outer = True
                    stop_reason = "converged_outer_value"
                    break

                value_prev_outer = tf.identity(value)

        value_3d = tf.reshape(value, [n_z, n_k, n_b])
        (
            r_tilde_grid,
            default_mask,
            expected_recovery,
            solvent_probability,
        ) = _compute_pricing_update_tf(
            env,
            prob_matrix,
            value_3d,
            recovery_flat,
            choice_b,
            n_z,
            n_k,
            n_b,
            dtype,
        )

        policy_action, policy_endo = _extract_policy_tf(
            env,
            choice_product,
            endo_product,
            policy_idx,
            dtype,
        )
        zero_profit_residual, funded_mask = _zero_profit_residual_tf(
            env,
            b_grid,
            r_tilde_grid,
            expected_recovery,
            solvent_probability,
            dtype,
        )

        env.install_r_tilde_grid(grids, tf.cast(r_tilde_grid, tf.float32).numpy())
        device_name = _normalize_device_name(value.device)

    return {
        "value": value,
        "policy_action": policy_action,
        "policy_endo": policy_endo,
        "grids": grids,
        "prob_matrix": prob_matrix,
        "r_tilde_grid": r_tilde_grid,
        "default_mask": default_mask,
        "zero_profit_residual": zero_profit_residual,
        "funded_mask": funded_mask,
        "converged_inner": converged_inner,
        "converged_outer": converged_outer,
        "n_outer": n_outer,
        "n_inner_last": n_inner_last,
        "inner_diff_history": inner_diff_history,
        "outer_value_diff_history": outer_value_diff_history,
        "pricing_diff_history": pricing_diff_history,
        "history": eval_history,
        "stop_reason": stop_reason,
        "backend": "tensorflow",
        "dtype": runtime_config.dtype,
        "device": device_name,
    }
