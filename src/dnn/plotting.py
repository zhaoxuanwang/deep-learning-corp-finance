"""
src/dnn/plotting.py

Matplotlib plotting helpers for DNN experiments.
Provides loss curves, policy slices, and scenario comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union


# =============================================================================
# LOSS CURVE PLOTS
# =============================================================================

def plot_loss_curve(
    history: Dict,
    keys: Optional[List[str]] = None,
    title: str = "Training Loss",
    figsize: Tuple[int, int] = (8, 5),
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot loss vs iteration for a single training run.
    
    Args:
        history: Training history dict with 'iteration' and loss keys
        keys: Loss keys to plot (default: auto-detect)
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale for y-axis
    
    Returns:
        Matplotlib Figure
    """
    if keys is None:
        keys = [k for k in history.keys() 
                if k.startswith("loss") and not k.startswith("_")]
    
    iterations = history.get("iteration", list(range(len(history.get(keys[0], [])))))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for key in keys:
        if key in history and len(history[key]) > 0:
            ax.plot(iterations, history[key], label=key.replace("_", " ").title())
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_loss_comparison(
    histories: List[Dict],
    labels: List[str],
    keys: Optional[List[str]] = None,
    title: str = "Loss Comparison",
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = False
) -> plt.Figure:
    """
    Overlay multiple training runs on the same axes.
    
    Args:
        histories: List of training history dicts
        labels: Label for each run
        keys: Loss keys to plot (default: common keys)
        title: Plot title
        figsize: Figure size
        log_scale: Use log scale
    
    Returns:
        Matplotlib Figure
    """
    if keys is None:
        # Find common loss keys
        all_keys = set()
        for h in histories:
            all_keys.update(k for k in h.keys() if k.startswith("loss"))
        keys = list(all_keys)
    
    n_keys = len(keys)
    if n_keys == 0:
        raise ValueError("No loss keys found in histories")
    
    fig, axes = plt.subplots(1, n_keys, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for key_idx, key in enumerate(keys):
        ax = axes[key_idx]
        
        for hist, label, color in zip(histories, labels, colors):
            if key in hist and len(hist[key]) > 0:
                iterations = hist.get("iteration", list(range(len(hist[key]))))
                ax.plot(iterations, hist[key], label=label, color=color)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.set_title(key.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_run_registry(
    registry: List[Dict],
    group_by: str = "label",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot runs from a registry (list of dicts with 'history' and 'label').
    
    Args:
        registry: List of {"history": dict, "label": str, ...}
        group_by: Key to group runs by
        figsize: Figure size
    
    Returns:
        Matplotlib Figure
    """
    histories = [r["history"] for r in registry]
    labels = [r.get("label", f"Run {i}") for i, r in enumerate(registry)]
    
    return plot_loss_comparison(histories, labels, figsize=figsize)


# =============================================================================
# ECONOMIC OBJECT PLOTS
# =============================================================================

def plot_policy_slice(
    eval_data: Dict,
    x_key: str,
    y_key: str,
    fixed_key: str,
    fixed_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot 1D slice of policy function.
    
    Args:
        eval_data: Output from evaluate_* functions
        x_key: Key for x-axis (e.g., "k" or "z")
        y_key: Key for y-axis (e.g., "k_next", "I_k")
        fixed_key: Key that is fixed (sliced)
        fixed_idx: Index along fixed dimension
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib Figure
    """
    x_data = eval_data[x_key]
    y_data = eval_data[y_key]
    
    # Determine slice axis
    if x_key == "k":
        x_slice = x_data[:, fixed_idx]
        y_slice = y_data[:, fixed_idx]
        fixed_val = eval_data.get(fixed_key, [[fixed_idx]])[0, fixed_idx]
    else:
        x_slice = x_data[fixed_idx, :]
        y_slice = y_data[fixed_idx, :]
        fixed_val = eval_data.get(fixed_key, [[fixed_idx]])[fixed_idx, 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_slice, y_slice, 'b-', linewidth=2)
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key.replace("_", " "))
    
    if title is None:
        title = f"{y_key} vs {x_key} (at {fixed_key}={fixed_val:.2f})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scenario_comparison(
    eval_datas: List[Dict],
    labels: List[str],
    x_key: str,
    y_key: str,
    fixed_key: str,
    fixed_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Overlay policy slices across scenarios.
    
    Args:
        eval_datas: List of evaluation outputs
        labels: Scenario labels
        x_key, y_key, fixed_key, fixed_idx: Same as plot_policy_slice
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_datas)))
    
    for data, label, color in zip(eval_datas, labels, colors):
        x_data = data[x_key]
        y_data = data[y_key]
        
        if x_key == "k":
            x_slice = x_data[:, fixed_idx]
            y_slice = y_data[:, fixed_idx]
        else:
            x_slice = x_data[fixed_idx, :]
            y_slice = y_data[fixed_idx, :]
        
        ax.plot(x_slice, y_slice, label=label, color=color, linewidth=2)
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key.replace("_", " "))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if title is None:
        title = f"{y_key} vs {x_key} Comparison"
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_2d_heatmap(
    eval_data: Dict,
    value_key: str,
    x_key: str = "k",
    y_key: str = "z",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "viridis"
) -> plt.Figure:
    """
    2D heatmap of policy/value function.
    
    Args:
        eval_data: Output from evaluate_* functions
        value_key: Key for values (e.g., "k_next", "V")
        x_key, y_key: Keys for axes
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    
    Returns:
        Matplotlib Figure
    """
    x_data = eval_data[x_key]
    y_data = eval_data[y_key]
    z_data = eval_data[value_key]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(x_data, y_data, z_data, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax, label=value_key.replace("_", " "))
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    
    if title is None:
        title = f"{value_key.replace('_', ' ')} Heatmap"
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MOMENTS TABLE DISPLAY
# =============================================================================

def display_moments_table(
    moments_df,
    title: str = "Summary Moments",
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Display moments DataFrame as a matplotlib table.
    
    Args:
        moments_df: DataFrame from compute_moments
        title: Table title
        figsize: Figure size
    
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Format numbers
    df_display = moments_df.copy()
    for col in df_display.select_dtypes(include=[np.number]).columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CONVENIENCE: QUICK PLOT FROM HISTORY
# =============================================================================

def quick_plot(
    history: Dict,
    what: str = "loss",
    **kwargs
) -> plt.Figure:
    """
    Quick plot from training history.
    
    Args:
        history: Training history dict
        what: "loss" for loss curves, "epsilon_D" for smoothing schedule
        **kwargs: Passed to underlying plot function
    
    Returns:
        Matplotlib Figure
    """
    if what == "loss":
        return plot_loss_curve(history, **kwargs)
    elif what == "epsilon_D" and "epsilon_D" in history:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
        iterations = history.get("iteration", list(range(len(history["epsilon_D"]))))
        ax.plot(iterations, history["epsilon_D"], 'b-', linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("epsilon_D")
        ax.set_title("Default Smoothing Schedule")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    else:
        return plot_loss_curve(history, **kwargs)
