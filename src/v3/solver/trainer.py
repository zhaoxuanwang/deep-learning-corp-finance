"""Block-1 policy-iteration training loop (DF26 Sec 3.4, Sec 8).

Each epoch runs ``steps_per_epoch`` gradient steps; the target network is reset to
the current value weights at the start of each epoch (Sec 3.5); the learning rate
decays 1% per epoch with a 1e-6 floor (Table A2). Written behind a thin function so
the controller/Engine (later increments) can drive it.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NET, make_adam
from src.v3.common.quadrature import gauss_hermite
from src.v3.solver.sampling import make_batch
from src.v3.solver.train_step import make_step_fn


def _set_lr(optimizer, lr) -> None:
    try:
        optimizer.learning_rate = lr
    except (AttributeError, ValueError):  # pragma: no cover - optimizer-version dependent
        tf.keras.backend.set_value(optimizer.learning_rate, lr)


def train_block1(bundle, bounds, ext, grid_cfg, train_cfg, master_seed, n_epochs,
                 *, steps_per_epoch=None, batch_size=None, eval_fn=None,
                 compile_step=True, verbose=False):
    """Train the value/policy networks; returns a per-epoch history list."""
    param_scaler = ParamScaler(bounds, dtype=TF_FLOAT_NET)
    x_nodes, w_nodes = gauss_hermite(train_cfg.gh_nodes, dtype=TF_FLOAT_NET)
    tau = tf.constant(train_cfg.smooth_tau, TF_FLOAT_NET)
    value_opt = make_adam(train_cfg.learning_rate)
    policy_opt = make_adam(train_cfg.learning_rate)
    step_fn = make_step_fn(bundle, ext, grid_cfg, x_nodes, w_nodes, tau,
                           value_opt, policy_opt, compile_step=compile_step)

    steps = steps_per_epoch or train_cfg.steps_per_epoch
    n = batch_size or train_cfg.batch_size
    history = []
    for epoch in range(n_epochs):
        bundle.sync_target()
        lr = max(train_cfg.learning_rate * (train_cfg.lr_decay ** epoch), train_cfg.lr_floor)
        _set_lr(value_opt, lr)
        _set_lr(policy_opt, lr)
        loss_v = loss_pi = 0.0
        for step in range(steps):
            batch = make_batch(n, bounds, ext, grid_cfg, param_scaler,
                               master_seed, epoch, step)
            lv, lp = step_fn(batch)
            loss_v, loss_pi = float(lv), float(lp)
        record = {"epoch": epoch, "loss_v": loss_v, "loss_pi": loss_pi, "lr": lr}
        if eval_fn is not None:
            extra = eval_fn(epoch, bundle)
            if extra:
                record.update(extra)
        history.append(record)
        if verbose:
            print(record)
    return history
