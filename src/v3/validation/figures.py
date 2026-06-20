"""Matplotlib helpers for the validation figures (notebooks only, DF26 Sec 12.3).

Kept thin: all logic lives in the library; these only render what is passed in.
"""
from __future__ import annotations

_TITLES = {"V": "Value V", "i": "Investment rate i",
           "bp": "Gross debt b'", "cp": "Cash c'"}

MOMENT_NAMES = ["Mean inv rate", "SD inv rate", "Mean op income", "SD op income",
                "Autocorr income", "Mean debt", "SD debt", "Mean cash", "SD cash",
                "Cash~net debt", "Cash~income"]
PARAM_NAMES = ["theta", "rho", "sigma", "delta", "gamma1", "gamma0", "chi", "cf"]


def _scatter_panels(ax_list, true, fitted, names, r2):
    import numpy as np
    for j, ax in enumerate(ax_list):
        x, y = true[:, j], fitted[:, j]
        ax.scatter(x, y, s=18, alpha=0.7)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, "k-", lw=1)
        ax.set_title(f"{names[j]}  (R^2={r2[j]:.2f})", fontsize=9)
        ax.set_xlabel("true"); ax.set_ylabel("fitted")


def plot_recovery_moments(out, savepath=None):
    """Fig V1 (DF26 Sec 12.2): true vs fitted moments, 11 panels."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    flat = axes.ravel()
    _scatter_panels(flat[:11], out["true_m"], out["fit_m"], MOMENT_NAMES, out["moment_r2"])
    flat[11].axis("off")
    fig.suptitle("Figure V1: true vs fitted moments")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_recovery_params(out, savepath=None):
    """Fig V2 (DF26 Sec 12.2): true vs fitted parameters, 8 panels."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    _scatter_panels(axes.ravel(), out["true_beta"], out["est_beta"], PARAM_NAMES, out["param_r2"])
    fig.suptitle("Figure V2: true vs fitted parameters")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_k_slices(slices, savepath=None):
    """Fig V3 headline: V/i/b'/c' vs k overlaying VFI / Network / Refined."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    k = slices["k"]
    for ax, key in zip(axes, ("V", "i", "bp", "cp")):
        d = slices[key]
        ax.plot(k, d["vfi"], "-", color="red", label="VFI")
        ax.plot(k, d["network"], "-", color="steelblue", label="Network")
        ax.plot(k, d["refined"], "--", color="seagreen", label="Refined")
        ax.set_xlabel("capital k")
        ax.set_title(_TITLES[key])
    axes[0].legend()
    fig.suptitle(f"Policy slices at z = {slices['z']:.2f}, b = {slices['b']:.2f}  "
                 f"(VFI / Network / Network Refined)")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
