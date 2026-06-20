"""Moment-surrogate networks (DF26 Sec 5.1-5.3).

One network per (moment, CV fold): M=11 moments x K=10 folds = 110 nets, each a
3x32 SiLU MLP mapping the normalized 8-dim parameter vector to one moment. Held as a
single batched ``tf.Module`` ([M, F, ...] weights). Trained continually (200 SGD
passes, Adam 1e-4, batch 256) on the (raw beta, moments) dataset; parameters are
normalized to [-1,1] with the current bounds at train time, so a bound change is a
scaler swap (Sec 5.3). Net (m, f) trains on rows whose fold != f; out-of-sample R^2
uses each row's held-out fold. float64 (feeds the float64 LM).
"""
from __future__ import annotations

import math

import tensorflow as tf

from src.v3.common import seeding
from src.v3.common.normalization import ParamScaler
from src.v3.common.precision import TF_FLOAT_NUM, make_adam
from src.v3.networks.film import silu

_EPS = 1e-12


class SurrogateEnsemble(tf.Module):
    def __init__(self, master_seed, n_moments=11, n_folds=10, hidden=32, layers=3,
                 param_dim=8, dtype=TF_FLOAT_NUM, name="surrogate"):
        super().__init__(name=name)
        self.M, self.F, self.layers, self.dtype_ = n_moments, n_folds, layers, dtype
        dims = [param_dim] + [hidden] * layers
        self.W = [self._glorot([n_moments, n_folds, dims[l], dims[l + 1]], master_seed, l, dtype)
                  for l in range(layers)]
        self.b = [tf.Variable(tf.zeros([n_moments, n_folds, dims[l + 1]], dtype)) for l in range(layers)]
        self.Wout = self._glorot([n_moments, n_folds, hidden, 1], master_seed, layers, dtype)
        self.bout = tf.Variable(tf.zeros([n_moments, n_folds, 1], dtype))
        self.fold = None        # row -> fold assignment (set in train)
        self.bounds = None

    @staticmethod
    def _glorot(shape, master, l, dtype):
        limit = math.sqrt(6.0 / (shape[-2] + shape[-1]))
        return tf.Variable(seeding.uniform(shape, master, seeding.Purpose.INIT, 100, l,
                                           minval=-limit, maxval=limit, dtype=dtype))

    def forward_all(self, beta_norm):
        """[B, 8] -> [B, M, F] predictions for every (moment, fold)."""
        h = silu(tf.einsum("bi,mfio->bmfo", beta_norm, self.W[0]) + self.b[0])
        for l in range(1, self.layers):
            h = silu(tf.einsum("bmfo,mfop->bmfp", h, self.W[l]) + self.b[l])
        return (tf.einsum("bmfo,mfoq->bmfq", h, self.Wout) + self.bout)[..., 0]

    def forward_fold(self, beta_norm, fold):
        """[B, 8] -> [B, M] predictions from fold ``fold``'s 11 nets (used by LM)."""
        h = silu(tf.einsum("bi,mio->bmo", beta_norm, self.W[0][:, fold]) + self.b[0][:, fold])
        for l in range(1, self.layers):
            h = silu(tf.einsum("bmo,mop->bmp", h, self.W[l][:, fold]) + self.b[l][:, fold])
        return (tf.einsum("bmo,moq->bmq", h, self.Wout[:, fold]) + self.bout[:, fold])[..., 0]

    def train(self, beta_raw, m_target, bounds, master_seed, *, passes=200, batch=256,
              lr=1e-4, max_obs=10000):
        self.bounds = bounds
        scaler = ParamScaler(bounds, dtype=self.dtype_)
        # Keep only the most-recent ``max_obs`` rows (Sec 5.3 / Table A2 10k cap). Rows
        # are appended over collection, so the tail is the most recent.
        beta_raw = tf.convert_to_tensor(beta_raw)
        m_target = tf.cast(m_target, self.dtype_)
        if beta_raw.shape[0] > max_obs:
            beta_raw, m_target = beta_raw[-max_obs:], m_target[-max_obs:]
        self.train_beta, self.train_m = beta_raw, m_target   # the rows the folds index
        beta_norm = scaler.normalize(beta_raw)                         # [N, 8]
        n = beta_norm.shape[0]
        self.fold = seeding.randint([n], master_seed, seeding.Purpose.FOLDS, minval=0, maxval=self.F)
        train_mask = tf.cast(self.fold[:, None] != tf.range(self.F)[None, :], self.dtype_)  # [N, F]
        opt = make_adam(lr)
        variables = self.trainable_variables

        def step(bn, mt, mk):
            with tf.GradientTape() as tape:
                pred = self.forward_all(bn)                            # [B, M, F]
                sq = (pred - mt[:, :, None]) ** 2                      # [B, M, F]
                per = tf.reduce_sum(mk[:, None, :] * sq, axis=0) / (tf.reduce_sum(mk, axis=0)[None, :] + _EPS)
                loss = tf.reduce_mean(per)
            opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return loss

        step = tf.function(step, reduce_retracing=True)
        for p in range(passes):
            perm = tf.random.experimental.stateless_shuffle(
                tf.range(n), seed=seeding.key(master_seed, seeding.Purpose.SURROGATE, p))
            for c in range(0, n, batch):
                idx = perm[c:c + batch]
                step(tf.gather(beta_norm, idx), tf.gather(m_target, idx), tf.gather(train_mask, idx))
        return self

    def oos_r2(self, beta_raw=None, m_target=None):
        """Per-moment out-of-sample R^2: each row predicted by its held-out fold (Sec 5.2).

        Defaults to the rows the surrogate was trained on (after the most-recent cap), so
        the fold assignment lines up; pass explicit rows only to score a different set.
        """
        if beta_raw is None:
            beta_raw, m_target = self.train_beta, self.train_m
        scaler = ParamScaler(self.bounds, dtype=self.dtype_)
        pred_all = self.forward_all(scaler.normalize(beta_raw))        # [N, M, F]
        oh = tf.one_hot(self.fold, self.F, dtype=self.dtype_)          # [N, F]
        pred = tf.einsum("nmf,nf->nm", pred_all, oh)                   # [N, M] held-out
        m_target = tf.cast(m_target, self.dtype_)
        ss_res = tf.reduce_sum((m_target - pred) ** 2, axis=0)
        ss_tot = tf.reduce_sum((m_target - tf.reduce_mean(m_target, axis=0)) ** 2, axis=0)
        return 1.0 - ss_res / (ss_tot + _EPS)
