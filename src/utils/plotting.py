"""
src/utils/plotting.py

Matplotlib plotting helpers for DNN experiments.
Provides loss curves, policy slices, and scenario comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union



# =============================================================================
# 2D POLICY SLICE PLOTTING
# =============================================================================

def plot_2d_policy_slice(
    eval_data: Dict,
    x_var: str,
    y_var: str,
    fixed_dim1_idx: Optional[int] = None,
    fixed_dim2_idx: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_45_line: bool = True,
    line_color: str = 'steelblue',
    line_style: str = '-',
    line_width: float = 2.0,
    label: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
    fix_axis_limits: bool = True
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot a 2D slice of policy function from evaluate_policy() output.

    Plots policy variable (k' or b') on y-axis against a state variable (k, z, or b)
    on x-axis, with other dimensions fixed. Optionally adds a 45-degree line for
    steady-state reference when x and y variables match (k' vs k, b' vs b).

    Args:
        eval_data: Output from evaluate_policy(). Must contain:
            - 'model_type': 'basic' or 'risky'
            - 'k_vals', 'z_vals', 'b_vals': 1D grid arrays
            - 'k_next', 'b_next': Policy outputs (2D or 3D grids)
            - 'fixed_*_idx': Default fixed indices
        x_var: State variable for x-axis. One of: 'k', 'z', 'logz', 'b'.
        y_var: Policy variable for y-axis. One of: 'k_next', 'b_next'.
        fixed_dim1_idx: Index for first fixed dimension (see dimension order below).
        fixed_dim2_idx: Index for second fixed dimension (risky model only).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Plot title. If None, auto-generated.
        show_45_line: If True, draw 45-degree line when x_var and y_var match
                      (k with k_next, b with b_next).
        line_color: Color for policy line.
        line_style: Line style for policy line.
        line_width: Line width for policy line.
        label: Label for legend.
        figsize: Figure size if creating new figure.
        fix_axis_limits: If True (default), enforce fixed axis limits based on
                         full state space bounds. X-axis uses the range of x_var,
                         Y-axis uses the range of corresponding state (k' → k bounds,
                         b' → b bounds).

    Dimension Order (for fixed_dim*_idx):
        - Basic model (k, z): x_var='k' fixes z (dim1), x_var='z' fixes k (dim1)
        - Risky model (k, z, b):
            x_var='k' fixes z (dim1) and b (dim2)
            x_var='z' fixes k (dim1) and b (dim2)
            x_var='b' fixes k (dim1) and z (dim2)

    Returns:
        If ax is None: plt.Figure
        If ax is provided: plt.Axes

    Example:
        >>> grid = evaluate_policy(result, k_bounds=(0.5, 3), logz_bounds=(-0.3, 0.3))
        >>> # Plot k' vs k at fixed z
        >>> fig = plot_2d_policy_slice(grid, x_var='k', y_var='k_next')
        >>> # Plot k' vs z at fixed k
        >>> fig = plot_2d_policy_slice(grid, x_var='z', y_var='k_next')
    """
    model_type = eval_data.get('model_type', 'basic')

    # Validate inputs
    valid_x_vars = ['k', 'z', 'logz', 'b'] if model_type == 'risky' else ['k', 'z', 'logz']
    valid_y_vars = ['k_next', 'b_next'] if model_type == 'risky' else ['k_next']

    if x_var not in valid_x_vars:
        raise ValueError(f"x_var must be one of {valid_x_vars}, got '{x_var}'")
    if y_var not in valid_y_vars:
        raise ValueError(f"y_var must be one of {valid_y_vars}, got '{y_var}'")

    # Extract 1D arrays and policy grids
    k_vals = eval_data['k_vals']
    z_vals = eval_data['z_vals']
    logz_vals = eval_data['logz_vals']
    b_vals = eval_data.get('b_vals')

    policy_grid = eval_data[y_var]
    if policy_grid is None:
        raise ValueError(f"y_var '{y_var}' is not available in eval_data")

    # Select x-axis values
    x_var_map = {'k': k_vals, 'z': z_vals, 'logz': logz_vals, 'b': b_vals}
    x_axis = x_var_map[x_var]

    # Use log(z) label when plotting logz
    x_label = 'log(z)' if x_var == 'logz' else x_var

    # Get fixed indices (use defaults from eval_data if not provided)
    fixed_k_idx = eval_data.get('fixed_k_idx', len(k_vals) // 2)
    fixed_z_idx = eval_data.get('fixed_z_idx', len(z_vals) // 2)
    fixed_b_idx = eval_data.get('fixed_b_idx', len(b_vals) // 2 if b_vals is not None else 0)

    # Extract 1D slice based on model type and x_var
    if model_type == 'basic':
        # Grid shape: (n_k, n_z)
        if x_var in ('k',):
            # Varying k, fix z
            idx = fixed_dim1_idx if fixed_dim1_idx is not None else fixed_z_idx
            y_slice = policy_grid[:, idx]
            fixed_val = z_vals[idx]
            fixed_name = 'z'
        else:  # z or logz
            # Varying z, fix k
            idx = fixed_dim1_idx if fixed_dim1_idx is not None else fixed_k_idx
            y_slice = policy_grid[idx, :]
            fixed_val = k_vals[idx]
            fixed_name = 'k'

    else:  # risky model
        # Grid shape: (n_k, n_z, n_b)
        if x_var == 'k':
            # Varying k, fix z and b
            z_idx = fixed_dim1_idx if fixed_dim1_idx is not None else fixed_z_idx
            b_idx = fixed_dim2_idx if fixed_dim2_idx is not None else fixed_b_idx
            y_slice = policy_grid[:, z_idx, b_idx]
            fixed_info = f"z={z_vals[z_idx]:.2f}, b={b_vals[b_idx]:.2f}"

        elif x_var in ('z', 'logz'):
            # Varying z, fix k and b
            k_idx = fixed_dim1_idx if fixed_dim1_idx is not None else fixed_k_idx
            b_idx = fixed_dim2_idx if fixed_dim2_idx is not None else fixed_b_idx
            y_slice = policy_grid[k_idx, :, b_idx]
            fixed_info = f"k={k_vals[k_idx]:.2f}, b={b_vals[b_idx]:.2f}"

        else:  # b
            # Varying b, fix k and z
            k_idx = fixed_dim1_idx if fixed_dim1_idx is not None else fixed_k_idx
            z_idx = fixed_dim2_idx if fixed_dim2_idx is not None else fixed_z_idx
            y_slice = policy_grid[k_idx, z_idx, :]
            fixed_info = f"k={k_vals[k_idx]:.2f}, z={z_vals[z_idx]:.2f}"

    # Create figure if ax not provided
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot policy line
    ax.plot(x_axis, y_slice, color=line_color, linestyle=line_style,
            linewidth=line_width, label=label)

    # Add 45-degree line if requested and variables match
    if show_45_line:
        # Check if x and y correspond (k with k_next, b with b_next)
        should_add_45 = (x_var == 'k' and y_var == 'k_next') or (x_var == 'b' and y_var == 'b_next')
        if should_add_45:
            ax.plot(x_axis, x_axis, 'k--', linewidth=1.0, alpha=0.7, label='45° (steady state)')

    # Set labels
    y_label = "k'" if y_var == 'k_next' else "b'"
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)

    # Generate title
    if title is None:
        if model_type == 'basic':
            title = f"{y_label} vs {x_label} (at {fixed_name}={fixed_val:.2f})"
        else:
            title = f"{y_label} vs {x_label} (at {fixed_info})"
    ax.set_title(title, fontsize=11)

    # Set fixed axis limits based on full state space bounds
    if fix_axis_limits:
        # X-axis: use full range of the x variable
        ax.set_xlim(x_axis.min(), x_axis.max())

        # Y-axis: use full range of corresponding state variable
        # k' should use k bounds, b' should use b bounds
        if y_var == 'k_next':
            ax.set_ylim(k_vals.min(), k_vals.max())
        elif y_var == 'b_next' and b_vals is not None:
            ax.set_ylim(b_vals.min(), b_vals.max())

    ax.grid(True, alpha=0.3)

    if label is not None or show_45_line:
        ax.legend(loc='best', fontsize=9)

    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_policy_panels(
    eval_data: Dict,
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    show_45_line: bool = True
) -> plt.Figure:
    """
    Create multi-panel plot of policy functions from evaluate_policy() output.

    Automatically creates appropriate panel layout based on model type:
    - Basic model: 2 panels (k' vs k, k' vs z)
    - Risky model: 2x3 panels (k' and b' against each of k, z, b)

    Args:
        eval_data: Output from evaluate_policy(). Must contain 'model_type'.
        figsize: Figure size. If None, auto-determined based on model type.
        suptitle: Super title for entire figure.
        show_45_line: If True, add 45-degree steady-state lines where applicable.

    Returns:
        plt.Figure: Multi-panel figure.

    Panel Layout:
        Basic Model (1x2):
            [k' vs k]  [k' vs log(z)]

        Risky Model (2x3):
            [k' vs k]  [k' vs log(z)]  [k' vs b]
            [b' vs k]  [b' vs log(z)]  [b' vs b]

    Example:
        >>> # Basic model
        >>> result = train_basic_br(dataset, ...)
        >>> grid = evaluate_policy(result, k_bounds=(0.5, 3), logz_bounds=(-0.3, 0.3))
        >>> fig = plot_policy_panels(grid, suptitle="Basic Model Policy")
        >>>
        >>> # Risky model
        >>> result = train_risky_br(dataset, ...)
        >>> grid = evaluate_policy(result, k_bounds=(0.5, 3), logz_bounds=(-0.3, 0.3),
        ...                        b_bounds=(0, 2))
        >>> fig = plot_policy_panels(grid, suptitle="Risky Debt Model Policy")
    """
    model_type = eval_data.get('model_type', 'basic')

    if model_type == 'basic':
        # 1x2 layout: k' vs k, k' vs z
        if figsize is None:
            figsize = (12, 5)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: k' vs k (at fixed z)
        plot_2d_policy_slice(
            eval_data, x_var='k', y_var='k_next',
            ax=axes[0], show_45_line=show_45_line
        )

        # Panel 2: k' vs log(z) (at fixed k)
        plot_2d_policy_slice(
            eval_data, x_var='logz', y_var='k_next',
            ax=axes[1], show_45_line=show_45_line
        )

    else:  # risky model
        # 2x3 layout
        if figsize is None:
            figsize = (15, 10)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Row 1: k' against k, log(z), b
        x_vars_row1 = ['k', 'logz', 'b']
        for col, x_var in enumerate(x_vars_row1):
            plot_2d_policy_slice(
                eval_data, x_var=x_var, y_var='k_next',
                ax=axes[0, col], show_45_line=show_45_line
            )

        # Row 2: b' against k, log(z), b
        x_vars_row2 = ['k', 'logz', 'b']
        for col, x_var in enumerate(x_vars_row2):
            plot_2d_policy_slice(
                eval_data, x_var=x_var, y_var='b_next',
                ax=axes[1, col], show_45_line=show_45_line
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_policy_comparison_panels(
    eval_datas: List[Dict],
    labels: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    show_45_line: bool = True
) -> plt.Figure:
    """
    Create multi-panel comparison plot overlaying multiple policies.

    Similar to plot_policy_panels() but overlays multiple policies on each panel
    for comparison across different training runs or parameter settings.

    Args:
        eval_datas: List of outputs from evaluate_policy().
        labels: Labels for each policy (for legend).
        figsize: Figure size. If None, auto-determined.
        suptitle: Super title for entire figure.
        show_45_line: If True, add 45-degree steady-state lines.

    Returns:
        plt.Figure: Multi-panel comparison figure.

    Example:
        >>> # Compare LR vs ER vs BR methods
        >>> grid_lr = evaluate_policy(result_lr, ...)
        >>> grid_er = evaluate_policy(result_er, ...)
        >>> grid_br = evaluate_policy(result_br, ...)
        >>> fig = plot_policy_comparison_panels(
        ...     [grid_lr, grid_er, grid_br],
        ...     labels=['LR', 'ER', 'BR'],
        ...     suptitle="Method Comparison"
        ... )
    """
    if len(eval_datas) == 0:
        raise ValueError("eval_datas must not be empty")

    # Use first dataset to determine model type
    model_type = eval_datas[0].get('model_type', 'basic')

    # Verify all have same model type
    for i, d in enumerate(eval_datas):
        if d.get('model_type', 'basic') != model_type:
            raise ValueError(f"All eval_datas must have same model_type. "
                           f"Index 0 is '{model_type}', index {i} is '{d.get('model_type')}'")

    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_datas)))

    if model_type == 'basic':
        if figsize is None:
            figsize = (12, 5)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: k' vs k
        for data, lbl, color in zip(eval_datas, labels, colors):
            plot_2d_policy_slice(
                data, x_var='k', y_var='k_next',
                ax=axes[0], show_45_line=False,
                line_color=color, label=lbl
            )
        if show_45_line:
            k_vals = eval_datas[0]['k_vals']
            axes[0].plot(k_vals, k_vals, 'k--', linewidth=1.0, alpha=0.7, label='45°')
        axes[0].legend(loc='best', fontsize=9)

        # Panel 2: k' vs log(z)
        for data, lbl, color in zip(eval_datas, labels, colors):
            plot_2d_policy_slice(
                data, x_var='logz', y_var='k_next',
                ax=axes[1], show_45_line=False,
                line_color=color, label=lbl
            )
        axes[1].legend(loc='best', fontsize=9)

    else:  # risky model
        if figsize is None:
            figsize = (15, 10)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        panel_configs = [
            # Row 0: k' vs k, log(z), b
            (0, 0, 'k', 'k_next'),
            (0, 1, 'logz', 'k_next'),
            (0, 2, 'b', 'k_next'),
            # Row 1: b' vs k, log(z), b
            (1, 0, 'k', 'b_next'),
            (1, 1, 'logz', 'b_next'),
            (1, 2, 'b', 'b_next'),
        ]

        for row, col, x_var, y_var in panel_configs:
            ax = axes[row, col]
            for data, lbl, color in zip(eval_datas, labels, colors):
                plot_2d_policy_slice(
                    data, x_var=x_var, y_var=y_var,
                    ax=ax, show_45_line=False,
                    line_color=color, label=lbl
                )

            # Add 45-degree line if applicable
            if show_45_line and ((x_var == 'k' and y_var == 'k_next') or
                                 (x_var == 'b' and y_var == 'b_next')):
                vals = eval_datas[0][f'{x_var}_vals']
                ax.plot(vals, vals, 'k--', linewidth=1.0, alpha=0.7, label='45°')

            ax.legend(loc='best', fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig
