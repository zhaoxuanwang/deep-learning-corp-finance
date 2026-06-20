"""Shared (parameter, moment) dataset buffer (DF26 Sec 5.3, 7).

A ring buffer holding the most-recent ``max_obs`` (raw beta, moments) rows. Collectors
append; the surrogate trains on the accumulated rows and keeps only the most recent
``max_obs`` (Table A2 10k cap), so the surrogates stay trained on the latest, best
collections as the trainer improves the network. In the async design this is the buffer
the collector GPUs write and the controller reads; here it is an in-process list.
"""
from __future__ import annotations

import tensorflow as tf

from src.v3.common.precision import TF_FLOAT_NUM


class DatasetBuffer:
    def __init__(self, max_obs=10000):
        self.max_obs = max_obs
        self._betas = None     # [N, 8]
        self._moments = None   # [N, 11]

    def add(self, beta_rows, moment_rows):
        """Append rows (each [b, .]) and truncate to the most-recent ``max_obs``."""
        beta_rows = tf.cast(beta_rows, TF_FLOAT_NUM)
        moment_rows = tf.cast(moment_rows, TF_FLOAT_NUM)
        if self._betas is None:
            self._betas, self._moments = beta_rows, moment_rows
        else:
            self._betas = tf.concat([self._betas, beta_rows], axis=0)
            self._moments = tf.concat([self._moments, moment_rows], axis=0)
        if self._betas.shape[0] > self.max_obs:
            self._betas = self._betas[-self.max_obs:]
            self._moments = self._moments[-self.max_obs:]
        return self

    def __len__(self):
        return 0 if self._betas is None else int(self._betas.shape[0])

    @property
    def betas(self):
        return self._betas

    @property
    def moments(self):
        return self._moments

    def recent(self, n):
        """The most-recent ``n`` (beta, moments) rows."""
        if self._betas is None:
            return self._betas, self._moments
        return self._betas[-n:], self._moments[-n:]
