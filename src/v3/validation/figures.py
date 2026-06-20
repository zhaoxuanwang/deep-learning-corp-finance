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


_EST_COLOR = "royalblue"   # single colour for the point estimate and its 95% CI spike


def _scatter_panels(ax_list, true, fitted, names, r2, yerr=None, ranges=None):
    """Per-panel true-vs-fitted scatter with a fixed-range 45-degree line and 95% CI spikes.

    Estimate markers and the capped CI spikes share one colour. Axes are fixed to ``ranges[j]``
    (``[lo, hi]`` per panel) when given, else to a common equal range from the data; x and y use
    the same limits with a square box, so the 45-degree line is visually diagonal and coverage is
    read off the full range. ``yerr`` is the CI half-width per point, shape [N, J]."""
    import numpy as np
    for j, ax in enumerate(ax_list):
        x, y = true[:, j], fitted[:, j]
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr[:, j], fmt="none", ecolor=_EST_COLOR, elinewidth=1.0,
                        capsize=2.5, capthick=1.0, alpha=0.6, zorder=2, label="95% CI")
        ax.scatter(x, y, s=14, c=_EST_COLOR, edgecolors="none", zorder=3, label="estimate")
        if ranges is not None:
            lo, hi = float(ranges[j][0]), float(ranges[j][1])
        else:
            parts = [x, y] + ([y - yerr[:, j], y + yerr[:, j]] if yerr is not None else [])
            allv = np.concatenate(parts)
            pad = 0.05 * (allv.max() - allv.min() + 1e-12)
            lo, hi = float(allv.min() - pad), float(allv.max() + pad)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1, label="45°")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
        ax.set_title(f"{names[j]}  (R² = {r2[j]:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("true"); ax.set_ylabel("fitted")


def _ci_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.02), fontsize=10, framealpha=0.9)


def plot_recovery_moments(out, savepath=None):
    """Fig V1 (DF26 Sec 12.2): true vs fitted moments, 11 panels, fixed-range with 95% CI spikes."""
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    flat = axes.ravel()
    se = out.get("fit_m_se")
    yerr = 1.96 * np.asarray(se) if se is not None and np.size(se) else None
    _scatter_panels(flat[:11], out["true_m"], out["fit_m"], MOMENT_NAMES, out["moment_r2"], yerr)
    flat[11].axis("off")
    _ci_legend(fig, flat[0])
    fig.suptitle("Figure V1: true vs fitted moments (R² = squared correlation; 95% CI: firm-clustered SE)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_recovery_params(out, savepath=None, bounds=None):
    """Fig V2 (DF26 Sec 12.2): true vs fitted parameters, 8 panels, axes fixed to the Table A1
    parameter ranges, with 95% CI spikes."""
    import matplotlib.pyplot as plt
    import numpy as np
    from src.v3.config import ParamBounds
    b = bounds if bounds is not None else ParamBounds.table_a1()
    ranges = list(zip(b.lower_array(), b.upper_array()))
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    folds = out.get("est_beta_folds")
    yerr = 1.96 * np.std(folds, axis=1) if folds is not None and np.size(folds) else None
    _scatter_panels(axes.ravel(), out["true_beta"], out["est_beta"], PARAM_NAMES,
                    out["param_r2"], yerr, ranges)
    _ci_legend(fig, axes.ravel()[0])
    fig.suptitle("Figure V2: true vs fitted parameters (R² = squared correlation; 95% CI: across-fold SD)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
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


# Shared line style: refined network drawn on top as a thick black dash; the raw VFI (green)
# and raw network (blue) are thinner and underneath, so the refined curve's tracking is visible.
_REFINED_KW = dict(color="black", lw=3.0, ls=(0, (5, 2)), zorder=3)
_VFI_KW = dict(color="green", lw=1.5, zorder=1)
_NET_KW = dict(color="royalblue", lw=1.5, zorder=2)
_LBL_VFI, _LBL_NET, _LBL_REF = "Raw VFI", "Raw Network", "Refined Network (Used)"


def _shared_bottom_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02),
               fontsize=10, frameon=True)


def plot_policy_slices(sl, savepath=None):
    """Fig V3 (DF26 Sec 12.3): V/i/b'/c' along one state axis. Refined network = thick black dash
    on top; raw VFI (green) and raw network (blue) thinner underneath. ``sl`` from
    ``policy_slices.slices(...)``."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    x, f = sl["coord"], sl["fixed"]
    for ax, key in zip(axes, ("V", "i", "bp", "cp")):
        d = sl[key]
        ax.plot(x, d["vfi"], label=_LBL_VFI, **_VFI_KW)
        ax.plot(x, d["network"], label=_LBL_NET, **_NET_KW)
        ax.plot(x, d["refined"], label=_LBL_REF, **_REFINED_KW)
        ax.set_xlabel(sl["axis_label"]); ax.set_title(_TITLES[key], fontweight="bold")
    held = {"k": f"z={f['z']:.2f}, b={f['b']:.2f}", "b": f"z={f['z']:.2f}, k={f['k']:.2f}",
            "z": f"k={f['k']:.2f}, b={f['b']:.2f}"}[sl["axis"]]
    fig.suptitle(f"Policy slices vs {sl['axis']}  ({held})", fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _shared_bottom_legend(fig, axes[0], ncol=3)
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_comparative_statics(cs, savepath=None):
    """Comparative statics (DF26 Sec 12.3): one parameter on x, a policy (i, b', c') on y, 1x3.
    Refined network = thick black dash on top; raw network (blue) thinner underneath; raw VFI is
    omitted (a VFI solve per swept point is infeasible). ``cs`` from
    ``comparative.comparative_statics(...)``."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = cs["values"]
    for ax, key in zip(axes, ("i", "bp", "cp")):
        d = cs[key]
        ax.plot(x, d["network"], label=_LBL_NET, **_NET_KW)
        ax.plot(x, d["refined"], label=_LBL_REF, **_REFINED_KW)
        ax.set_xlabel(cs["param"]); ax.set_title(_TITLES[key], fontweight="bold")
    fig.suptitle(f"Comparative statics vs {cs['param']}", fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _shared_bottom_legend(fig, axes[0], ncol=2)
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
