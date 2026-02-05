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
    show_45_line: bool = True,
    frictionless_benchmark: Optional[Dict] = None
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
        frictionless_benchmark: Optional dict with 'params' and 'shock_params'.
            If provided, overlays the analytical frictionless solution as a
            dashed black line. Only applies to basic model.

    Returns:
        plt.Figure: Multi-panel comparison figure.

    Example:
        >>> # Compare LR vs ER vs BR methods
        >>> fig = plot_policy_comparison_panels(
        ...     [grid_lr, grid_er, grid_br],
        ...     labels=['LR', 'ER', 'BR'],
        ...     suptitle="Method Comparison"
        ... )
        >>> # With frictionless benchmark (for baseline scenario)
        >>> fig = plot_policy_comparison_panels(
        ...     [grid_lr, grid_er, grid_br],
        ...     labels=['LR', 'ER', 'BR'],
        ...     frictionless_benchmark={'params': params, 'shock_params': shock_params}
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

        # Add frictionless benchmark (horizontal line at fixed z) - drawn on top
        if frictionless_benchmark is not None:
            from src.utils.analysis import compute_frictionless_policy
            params = frictionless_benchmark['params']
            shock_params = frictionless_benchmark['shock_params']
            fixed_z = eval_datas[0]['fixed_z_val']
            k_star_frictionless = compute_frictionless_policy(fixed_z, params, shock_params)
            axes[0].axhline(y=k_star_frictionless, color='black', linestyle='--',
                           linewidth=2.0, alpha=1.0, label='Frictionless', zorder=10)
        axes[0].legend(loc='best', fontsize=9)

        # Panel 2: k' vs log(z)
        for data, lbl, color in zip(eval_datas, labels, colors):
            plot_2d_policy_slice(
                data, x_var='logz', y_var='k_next',
                ax=axes[1], show_45_line=False,
                line_color=color, label=lbl
            )

        # Add frictionless benchmark curve - drawn on top
        if frictionless_benchmark is not None:
            from src.utils.analysis import compute_frictionless_policy
            params = frictionless_benchmark['params']
            shock_params = frictionless_benchmark['shock_params']
            z_vals = eval_datas[0]['z_vals']
            logz_vals = eval_datas[0]['logz_vals']
            k_star_curve = compute_frictionless_policy(z_vals, params, shock_params)
            axes[1].plot(logz_vals, k_star_curve, 'k--', linewidth=2.0,
                        alpha=1.0, label='Frictionless', zorder=10)
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


# =============================================================================
# 3D POLICY SURFACE PLOTTING
# =============================================================================

def plot_3d_policy_slice(
    eval_data: Dict,
    y_var: str = 'k_next',
    fixed_dim_idx: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'coolwarm',
    alpha: float = 0.85,
    figsize: Tuple[int, int] = (8, 6),
    elev: float = 25,
    azim: float = -60,
    show_wireframe: bool = False,
    show_colorbar: bool = True,
    show_contour_projection: bool = False,
    linewidth: float = 0.1,
    antialiased: bool = True,
    fix_axis_limits: bool = True
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot a 3D surface of policy function from evaluate_policy() output.

    Creates a 3D surface visualization where the policy output (k' or b')
    is plotted as height over a 2D plane of state variables. Uses z (not log(z))
    on Y-axis so the origin is easy to interpret since z > 0.

    The fixed dimension index uses the steady-state value stored in eval_data
    (from get_steady_state_policy → evaluate_policy workflow).

    Args:
        eval_data: Output from evaluate_policy(). Must contain:
            - 'model_type': 'basic' or 'risky'
            - 'k_vals', 'z_vals', 'b_vals': 1D grid arrays
            - 'k_next', 'b_next': Policy outputs (2D or 3D grids)
            - 'fixed_k_idx', 'fixed_b_idx', 'fixed_z_idx': Steady-state indices
        y_var: Policy variable for z-axis. One of: 'k_next', 'b_next'.
            - 'k_next': Plot k' over (k, z) plane
            - 'b_next': Plot b' over (b, z) plane (risky model only)
        fixed_dim_idx: Index for fixed dimension (risky model only).
            - For y_var='k_next': fixes b dimension (default: fixed_b_idx from SS)
            - For y_var='b_next': fixes k dimension (default: fixed_k_idx from SS)
        ax: Matplotlib 3D axes to plot on. If None, creates new figure.
            Must be created with projection='3d'.
        title: Plot title. If None, auto-generated.
        cmap: Colormap name for surface coloring (default: 'coolwarm').
        alpha: Surface transparency (0-1).
        figsize: Figure size (width, height) if creating new figure.
        elev: Elevation angle in degrees for 3D view.
        azim: Azimuth angle in degrees for 3D view.
        show_wireframe: If True, show wireframe overlay on surface.
        show_colorbar: If True, add colorbar showing z-values.
        show_contour_projection: If True, project contour lines onto xy-plane.
        linewidth: Width of surface edge lines.
        antialiased: If True, use antialiasing for smoother rendering.
        fix_axis_limits: If True (default), enforce fixed axis limits based on
                         full state space bounds for consistency across plots.

    Axis Mapping:
        Basic model:
            X-axis: k (capital), Y-axis: z (productivity), Z-axis: k'

        Risky model (y_var='k_next'):
            X-axis: k, Y-axis: z, Z-axis: k' (at fixed b=b*)

        Risky model (y_var='b_next'):
            X-axis: b, Y-axis: z, Z-axis: b' (at fixed k=k*)

    Returns:
        If ax is None: plt.Figure containing the 3D plot
        If ax is provided: plt.Axes (the same ax, now with plot)

    Example:
        >>> # Workflow: get steady state, evaluate policy, then plot
        >>> ss = get_steady_state_policy(result, k_bounds, logz_bounds, b_bounds)
        >>> grid = evaluate_policy(result, k_bounds, logz_bounds, b_bounds,
        ...                        fixed_k_val=ss['k_star_val'],
        ...                        fixed_b_val=ss['b_star_val'])
        >>> fig = plot_3d_policy_slice(grid, y_var='k_next')
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    model_type = eval_data.get('model_type', 'basic')

    # Validate y_var
    valid_y_vars = ['k_next', 'b_next'] if model_type == 'risky' else ['k_next']
    if y_var not in valid_y_vars:
        raise ValueError(f"y_var must be one of {valid_y_vars}, got '{y_var}'")

    # Extract grid arrays
    k_vals = eval_data['k_vals']
    z_vals = eval_data['z_vals']  # Use z (not logz) for interpretability
    b_vals = eval_data.get('b_vals')

    policy_grid = eval_data[y_var]
    if policy_grid is None:
        raise ValueError(f"y_var '{y_var}' is not available in eval_data")

    # Get fixed indices from eval_data (set by evaluate_policy using SS values)
    fixed_k_idx = eval_data.get('fixed_k_idx', len(k_vals) // 2)
    fixed_z_idx = eval_data.get('fixed_z_idx', len(z_vals) // 2)
    fixed_b_idx = eval_data.get('fixed_b_idx', len(b_vals) // 2 if b_vals is not None else 0)

    # Extract 2D slice and determine axis labels
    if model_type == 'basic':
        # Grid shape: (n_k, n_z) - no slicing needed
        Z_surface = policy_grid
        x_vals, y_vals = k_vals, z_vals
        x_label, y_label, z_label = 'k', 'z', "k'"
        fixed_info = None
        # For z-axis limits, k' uses k bounds
        z_bounds_vals = k_vals
    else:  # risky model
        if y_var == 'k_next':
            # Plot k' over (k, z), fix b at steady state
            b_idx = fixed_dim_idx if fixed_dim_idx is not None else fixed_b_idx
            Z_surface = policy_grid[:, :, b_idx]
            x_vals, y_vals = k_vals, z_vals
            x_label, y_label, z_label = 'k', 'z', "k'"
            fixed_info = f"b={b_vals[b_idx]:.2f}"
            z_bounds_vals = k_vals
        else:  # y_var == 'b_next'
            # Plot b' over (b, z), fix k at steady state
            k_idx = fixed_dim_idx if fixed_dim_idx is not None else fixed_k_idx
            Z_surface = policy_grid[k_idx, :, :].T  # (n_b, n_z) after transpose
            x_vals, y_vals = b_vals, z_vals
            x_label, y_label, z_label = 'b', 'z', "b'"
            fixed_info = f"k={k_vals[k_idx]:.2f}"
            z_bounds_vals = b_vals

    # Create meshgrid (indexing='ij' matches (n_x, n_y) array layout)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z_surface, cmap=cmap, alpha=alpha,
        linewidth=linewidth, antialiased=antialiased,
        edgecolor='gray' if linewidth > 0 else 'none'
    )

    # Optional wireframe overlay
    if show_wireframe:
        ax.plot_wireframe(X, Y, Z_surface, color='black', linewidth=0.3, alpha=0.3)

    # Optional contour projection onto xy-plane
    if show_contour_projection:
        z_min = Z_surface.min()
        z_range = Z_surface.max() - z_min
        offset = z_min - 0.1 * z_range if z_range > 0 else z_min - 0.1
        ax.contour(X, Y, Z_surface, zdir='z', offset=offset,
                   cmap=cmap, alpha=0.5, levels=10)

    # Set labels
    ax.set_xlabel(x_label, fontsize=10, labelpad=8)
    ax.set_ylabel(y_label, fontsize=10, labelpad=8)
    ax.set_zlabel(z_label, fontsize=10, labelpad=8)

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Set fixed axis limits based on state space bounds
    if fix_axis_limits:
        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.set_ylim(y_vals.min(), y_vals.max())
        ax.set_zlim(z_bounds_vals.min(), z_bounds_vals.max())

    # Generate title
    if title is None:
        if model_type == 'basic':
            title = f"{z_label}({x_label}, {y_label}) Surface"
        else:
            title = f"{z_label}({x_label}, {y_label}) at {fixed_info}"
    ax.set_title(title, fontsize=11, pad=10)

    # Add colorbar
    if show_colorbar and created_fig:
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1, label=z_label)

    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_3d_policy_panels(
    eval_data: Dict,
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    cmap: str = 'coolwarm',
    elev: float = 25,
    azim: float = -60,
    show_colorbar: bool = True,
    show_contour_projection: bool = False,
    fix_axis_limits: bool = True
) -> plt.Figure:
    """
    Create multi-panel 3D surface plots of policy functions.

    Automatically creates appropriate panel layout based on model type:
    - Basic model: 1 panel showing k'(k, z)
    - Risky model: 2 panels showing k'(k, z) at b* and b'(b, z) at k*

    Uses steady-state indices stored in eval_data (from the standard workflow:
    get_steady_state_policy → evaluate_policy with fixed_k_val/fixed_b_val).

    Args:
        eval_data: Output from evaluate_policy(). Must contain 'model_type'
                   and fixed indices (fixed_k_idx, fixed_b_idx).
        figsize: Figure size. If None, auto-determined based on model type.
        suptitle: Super title for entire figure.
        cmap: Colormap for surfaces (default: 'coolwarm').
        elev: Elevation angle for 3D view.
        azim: Azimuth angle for 3D view.
        show_colorbar: If True, add colorbars to each panel.
        show_contour_projection: If True, add contour lines projected on xy-plane.
        fix_axis_limits: If True (default), use fixed state space bounds.

    Returns:
        plt.Figure: Multi-panel figure with 3D surfaces.

    Panel Layout:
        Basic Model (1 panel):
            [k'(k, z)]

        Risky Model (1x2):
            [k'(k, z) at b=b*]  [b'(b, z) at k=k*]

    Example:
        >>> # Basic model workflow
        >>> ss = get_steady_state_policy(result, k_bounds, logz_bounds)
        >>> grid = evaluate_policy(result, k_bounds, logz_bounds,
        ...                        fixed_k_val=ss['k_star_val'])
        >>> fig = plot_3d_policy_panels(grid, suptitle="Basic Model Policy Surface")
        >>>
        >>> # Risky model workflow
        >>> ss = get_steady_state_policy(result, k_bounds, logz_bounds, b_bounds)
        >>> grid = evaluate_policy(result, k_bounds, logz_bounds, b_bounds,
        ...                        fixed_k_val=ss['k_star_val'],
        ...                        fixed_b_val=ss['b_star_val'])
        >>> fig = plot_3d_policy_panels(grid, suptitle="Risky Debt Policies")
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    model_type = eval_data.get('model_type', 'basic')

    if model_type == 'basic':
        # Single panel: k'(k, z)
        if figsize is None:
            figsize = (8, 6)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        plot_3d_policy_slice(
            eval_data, y_var='k_next', ax=ax,
            cmap=cmap, elev=elev, azim=azim,
            show_colorbar=False,
            show_contour_projection=show_contour_projection,
            fix_axis_limits=fix_axis_limits
        )

        if show_colorbar:
            # Get the surface from plot_surface (first PolyCollection)
            mappable = ax.collections[0] if ax.collections else None
            if mappable:
                fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=15, pad=0.1, label="k'")

    else:  # risky model
        # Two panels: k'(k,z) at b*, b'(b,z) at k*
        if figsize is None:
            figsize = (14, 6)

        fig = plt.figure(figsize=figsize)

        # Panel 1: k'(k, z) at fixed b* (steady state debt)
        ax1 = fig.add_subplot(121, projection='3d')
        plot_3d_policy_slice(
            eval_data, y_var='k_next', ax=ax1,
            cmap=cmap, elev=elev, azim=azim,
            show_colorbar=False,
            show_contour_projection=show_contour_projection,
            fix_axis_limits=fix_axis_limits
        )

        # Panel 2: b'(b, z) at fixed k* (steady state capital)
        ax2 = fig.add_subplot(122, projection='3d')
        plot_3d_policy_slice(
            eval_data, y_var='b_next', ax=ax2,
            cmap=cmap, elev=elev, azim=azim,
            show_colorbar=False,
            show_contour_projection=show_contour_projection,
            fix_axis_limits=fix_axis_limits
        )

        # Add colorbars
        if show_colorbar:
            for ax, label in [(ax1, "k'"), (ax2, "b'")]:
                mappable = ax.collections[0] if ax.collections else None
                if mappable:
                    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=12, pad=0.12, label=label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# =============================================================================
# SCENARIO COMPARISON PLOTTING (Investment Rate & Leverage)
# =============================================================================

def _compute_investment_rate(k_next: np.ndarray, k_vals: np.ndarray, delta: float) -> np.ndarray:
    """
    Compute investment rate I/k = (k' - (1-δ)k) / k = k'/k - (1-δ).

    Args:
        k_next: Policy output k' (can be 1D, 2D, or 3D array)
        k_vals: 1D array of k grid values (broadcast along first axis)
        delta: Depreciation rate

    Returns:
        Investment rate array with same shape as k_next
    """
    # Reshape k_vals for broadcasting: (n_k,) -> (n_k, 1) or (n_k, 1, 1)
    if k_next.ndim == 1:
        k_broadcast = k_vals
    elif k_next.ndim == 2:
        k_broadcast = k_vals[:, np.newaxis]
    else:  # 3D
        k_broadcast = k_vals[:, np.newaxis, np.newaxis]

    return k_next / k_broadcast - (1 - delta)


def _compute_leverage_ratio(b_next: np.ndarray, k_vals: np.ndarray) -> np.ndarray:
    """
    Compute leverage ratio b'/k.

    Args:
        b_next: Policy output b' (3D array: n_k x n_z x n_b)
        k_vals: 1D array of k grid values

    Returns:
        Leverage ratio array with same shape as b_next
    """
    # Reshape k_vals for broadcasting: (n_k,) -> (n_k, 1, 1)
    k_broadcast = k_vals[:, np.newaxis, np.newaxis]
    return b_next / k_broadcast


def plot_scenario_comparison_panels(
    scenario_eval_datas: Dict[str, Dict],
    delta: float,
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    show_45_line: bool = False
) -> plt.Figure:
    """
    Create multi-panel comparison plot of transformed policies across scenarios.

    Compares investment rate (I/k) and leverage ratio (b'/k) across different
    economic scenarios (e.g., baseline, smooth_cost, fixed_cost) for a single
    training method (e.g., BR).

    Transformations:
        - Investment rate: I/k = (k' - (1-δ)k) / k = k'/k - (1-δ)
        - Leverage ratio: b'/k (risky model only)

    Args:
        scenario_eval_datas: Dict mapping scenario names to evaluate_policy() outputs.
            Example: {'baseline': grid_baseline, 'smooth_cost': grid_smooth, ...}
            Each eval_data must contain the policy grids and fixed indices.
        delta: Depreciation rate for investment rate calculation.
        figsize: Figure size. If None, auto-determined based on model type.
        suptitle: Super title for entire figure.
        show_45_line: If True, add reference lines where applicable.

    Returns:
        plt.Figure: Multi-panel comparison figure.

    Panel Layout:
        Basic Model (1x2):
            [I/k vs k]  [I/k vs log(z)]

        Risky Model (2x3):
            [I/k vs k]    [I/k vs log(z)]    [I/k vs b]
            [b'/k vs k]   [b'/k vs log(z)]   [b'/k vs b]

    Note:
        Each scenario uses its own steady-state indices (from evaluate_policy).
        The x-axis range is determined by the first scenario's grid.

    Example:
        >>> # Compare scenarios for BR method
        >>> grids = {
        ...     'baseline': evaluate_policy(results['baseline']['br'], ...),
        ...     'smooth_cost': evaluate_policy(results['smooth_cost']['br'], ...),
        ...     'fixed_cost': evaluate_policy(results['fixed_cost']['br'], ...)
        ... }
        >>> fig = plot_scenario_comparison_panels(
        ...     grids, delta=0.15,
        ...     suptitle="BR Method: Scenario Comparison"
        ... )
    """
    if len(scenario_eval_datas) == 0:
        raise ValueError("scenario_eval_datas must not be empty")

    # Get scenario names and eval_datas
    scenario_names = list(scenario_eval_datas.keys())
    eval_datas = list(scenario_eval_datas.values())

    # Use first dataset to determine model type and grid
    first_data = eval_datas[0]
    model_type = first_data.get('model_type', 'basic')

    # Verify all have same model type
    for name, data in scenario_eval_datas.items():
        if data.get('model_type', 'basic') != model_type:
            raise ValueError(f"All eval_datas must have same model_type. "
                           f"First is '{model_type}', '{name}' is '{data.get('model_type')}'")

    # Extract common grid (use first scenario's grid for x-axis)
    k_vals = first_data['k_vals']
    z_vals = first_data['z_vals']
    logz_vals = first_data['logz_vals']
    b_vals = first_data.get('b_vals')

    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))

    if model_type == 'basic':
        # 1x2 layout: I/k vs k, I/k vs log(z)
        if figsize is None:
            figsize = (12, 5)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: I/k vs k (at fixed z, using each scenario's fixed_z_idx)
        ax = axes[0]
        for data, name, color in zip(eval_datas, scenario_names, colors):
            k_next_grid = data['k_next']
            fixed_z_idx = data.get('fixed_z_idx', len(data['z_vals']) // 2)

            # Extract slice at fixed z
            k_next_slice = k_next_grid[:, fixed_z_idx]

            # Compute I/k
            inv_rate = _compute_investment_rate(k_next_slice, data['k_vals'], delta)

            ax.plot(data['k_vals'], inv_rate, color=color, linewidth=2.0, label=name)

        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('I/k', fontsize=10)
        ax.set_title("Investment Rate vs Capital (at z=1)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        if show_45_line:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

        # Panel 2: I/k vs log(z) (at fixed k, using each scenario's fixed_k_idx)
        ax = axes[1]
        for data, name, color in zip(eval_datas, scenario_names, colors):
            k_next_grid = data['k_next']
            fixed_k_idx = data.get('fixed_k_idx', len(data['k_vals']) // 2)

            # Extract slice at fixed k (steady state)
            k_next_slice = k_next_grid[fixed_k_idx, :]

            # For I/k at fixed k, use the fixed k value
            k_fixed = data['k_vals'][fixed_k_idx]
            inv_rate = k_next_slice / k_fixed - (1 - delta)

            ax.plot(data['logz_vals'], inv_rate, color=color, linewidth=2.0, label=name)

        ax.set_xlabel('log(z)', fontsize=10)
        ax.set_ylabel('I/k', fontsize=10)
        ax.set_title("Investment Rate vs Productivity (at k=k*)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        if show_45_line:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    else:  # risky model
        # 2x3 layout
        if figsize is None:
            figsize = (15, 10)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Row 0: I/k against k, log(z), b
        # Row 1: b'/k against k, log(z), b

        panel_configs = [
            # (row, col, x_var, y_transform, x_label, y_label, title_suffix)
            (0, 0, 'k', 'inv_rate', 'k', 'I/k', "vs Capital (at z=1, b=b*)"),
            (0, 1, 'logz', 'inv_rate', 'log(z)', 'I/k', "vs Productivity (at k=k*, b=b*)"),
            (0, 2, 'b', 'inv_rate', 'b', 'I/k', "vs Debt (at k=k*, z=1)"),
            (1, 0, 'k', 'leverage', 'k', "b'/k", "vs Capital (at z=1, b=b*)"),
            (1, 1, 'logz', 'leverage', 'log(z)', "b'/k", "vs Productivity (at k=k*, b=b*)"),
            (1, 2, 'b', 'leverage', 'b', "b'/k", "vs Debt (at k=k*, z=1)"),
        ]

        for row, col, x_var, y_transform, x_label, y_label, title_suffix in panel_configs:
            ax = axes[row, col]

            for data, name, color in zip(eval_datas, scenario_names, colors):
                # Get grid arrays from this scenario
                k_vals_s = data['k_vals']
                z_vals_s = data['z_vals']
                logz_vals_s = data['logz_vals']
                b_vals_s = data['b_vals']

                # Get fixed indices (scenario-specific steady state)
                fixed_k_idx = data.get('fixed_k_idx', len(k_vals_s) // 2)
                fixed_z_idx = data.get('fixed_z_idx', len(z_vals_s) // 2)
                fixed_b_idx = data.get('fixed_b_idx', len(b_vals_s) // 2)

                # Select appropriate policy grid
                if y_transform == 'inv_rate':
                    policy_grid = data['k_next']
                else:  # leverage
                    policy_grid = data['b_next']

                # Extract 1D slice based on x_var
                if x_var == 'k':
                    # Varying k, fix z and b
                    y_slice = policy_grid[:, fixed_z_idx, fixed_b_idx]
                    x_axis = k_vals_s

                    if y_transform == 'inv_rate':
                        # I/k = k'/k - (1-δ)
                        y_vals = y_slice / k_vals_s - (1 - delta)
                    else:  # leverage: b'/k
                        y_vals = y_slice / k_vals_s

                elif x_var == 'logz':
                    # Varying z, fix k and b
                    y_slice = policy_grid[fixed_k_idx, :, fixed_b_idx]
                    x_axis = logz_vals_s
                    k_fixed = k_vals_s[fixed_k_idx]

                    if y_transform == 'inv_rate':
                        y_vals = y_slice / k_fixed - (1 - delta)
                    else:  # leverage
                        y_vals = y_slice / k_fixed

                else:  # x_var == 'b'
                    # Varying b, fix k and z
                    y_slice = policy_grid[fixed_k_idx, fixed_z_idx, :]
                    x_axis = b_vals_s
                    k_fixed = k_vals_s[fixed_k_idx]

                    if y_transform == 'inv_rate':
                        y_vals = y_slice / k_fixed - (1 - delta)
                    else:  # leverage
                        y_vals = y_slice / k_fixed

                ax.plot(x_axis, y_vals, color=color, linewidth=2.0, label=name)

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f"{y_label} {title_suffix}", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

            if show_45_line:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# =============================================================================
# LOSS CURVE PLOTTING
# =============================================================================

def plot_basic_loss_curves(
    result_lr: Dict,
    result_er: Dict,
    result_br: Dict,
    scenario_name: str = "scenario",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot loss curves for all three basic model training methods.

    Creates a 2x3 panel figure:
        Row 1: LR loss, ER loss, BR Actor loss
        Row 2: LR (log scale), ER (log scale), BR Critic (rel_mse)

    Args:
        result_lr: Training result dict from LR method with 'history' key
        result_er: Training result dict from ER method with 'history' key
        result_br: Training result dict from BR method with 'history' key
        scenario_name: Name for figure title and filename
        save_path: If provided, save figure to this path
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Extract iteration arrays
    iter_lr = result_lr['history'].get('iteration', list(range(len(result_lr['history']['loss_LR']))))
    iter_er = result_er['history'].get('iteration', list(range(len(result_er['history']['loss_ER']))))
    iter_br = result_br['history'].get('iteration', list(range(len(result_br['history']['loss_actor']))))

    # Row 1: Primary loss curves

    # LR Loss (more negative = better)
    axes[0, 0].plot(iter_lr, result_lr['history']['loss_LR'],
                    color='blue', linewidth=2, label='LR Loss')
    axes[0, 0].set_title('LR: Lifetime Reward', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss (more negative = better)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # ER Loss (should approach 0)
    axes[0, 1].plot(iter_er, result_er['history']['loss_ER'],
                    color='orange', linewidth=2, label='ER Loss')
    axes[0, 1].set_title('ER: Euler Residual', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss (→ 0)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # BR Actor Loss (more negative = better)
    axes[0, 2].plot(iter_br, result_br['history']['loss_actor'],
                    color='green', linewidth=2, label='Actor Loss')
    axes[0, 2].set_title('BR: Actor Loss', fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Loss (more negative = better)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # Row 2: Log scale and relative metrics

    # LR Loss (log scale of absolute value)
    lr_loss = np.array(result_lr['history']['loss_LR'])
    axes[1, 0].plot(iter_lr, -lr_loss, color='blue', linewidth=2, label='-LR Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('LR: -Loss (log scale)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('-Loss (log scale)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # ER Loss (log scale)
    er_loss = np.array(result_er['history']['loss_ER'])
    # Handle potential negative values from cross-product
    er_loss_plot = np.maximum(np.abs(er_loss), 1e-10)
    axes[1, 1].plot(iter_er, er_loss_plot, color='orange', linewidth=2, label='|ER Loss|')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('ER: |Loss| (log scale)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('|Loss| (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # BR Critic: Relative MSE (should decrease)
    if 'rel_mse' in result_br['history']:
        axes[1, 2].plot(iter_br, result_br['history']['rel_mse'],
                        color='red', linewidth=2, label='Relative MSE')
        axes[1, 2].set_title('BR: Critic Rel MSE (should ↓)', fontsize=11, fontweight='bold')
        axes[1, 2].set_ylabel('Relative MSE')
    else:
        axes[1, 2].plot(iter_br, result_br['history']['loss_critic'],
                        color='red', linewidth=2, label='Critic Loss')
        axes[1, 2].set_title('BR: Critic Loss', fontsize=11, fontweight='bold')
        axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    fig.suptitle(f'Basic Model Loss Curves: {scenario_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_risky_loss_curves(
    result_risky: Dict,
    scenario_name: str = "baseline",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9)
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Plot loss curves for risky debt BR training.

    Creates a 2x2 panel figure:
        (0,0): Critic Relative MSE (should decrease)
        (0,1): Actor Loss (more negative = better)
        (1,0): Price Loss (should decrease)
        (1,1): Value Scale Growth (context)

    Args:
        result_risky: Training result dict with 'history' key
        scenario_name: Name for figure title
        save_path: If provided, save figure to this path
        figsize: Figure size (width, height)

    Returns:
        Tuple of (Figure, summary_dict) where summary_dict contains final metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    history = result_risky['history']
    iter_risky = history.get('iteration', list(range(len(history['loss_actor']))))

    # 1.1 Critic Loss - Relative MSE (should decrease)
    if 'rel_mse' in history:
        axes[0, 0].plot(iter_risky, history['rel_mse'],
                        color='red', linewidth=2, label='Relative MSE')
        axes[0, 0].set_title('Critic: Relative MSE (should ↓)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Relative MSE')
    else:
        axes[0, 0].plot(iter_risky, history['loss_critic'],
                        color='red', linewidth=2, label='Critic Loss')
        axes[0, 0].set_title('Critic Loss', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 1.2 Actor Loss (should become more negative)
    axes[0, 1].plot(iter_risky, history['loss_actor'],
                    color='green', linewidth=2, label='Actor Loss')
    axes[0, 1].set_title('Actor Loss (more negative = better)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 2.1 Price Loss (should decrease)
    axes[1, 0].plot(iter_risky, history['loss_price'],
                    color='purple', linewidth=2, label='Price Loss')
    axes[1, 0].set_title('Price Loss (Bond Pricing Residual)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 2.2 Value Scale Growth (context for MSE interpretation)
    if 'mean_value_scale' in history:
        axes[1, 1].plot(iter_risky, history['mean_value_scale'],
                        color='blue', linewidth=2, label='Mean |V|')
        axes[1, 1].set_title('Value Scale Growth', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Mean |V|')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Value scale not available',
                        ha='center', va='center', fontsize=12, color='gray',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Value Scale')

    fig.suptitle(f'Risky Debt BR Loss Curves: {scenario_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Compute summary metrics
    summary = {
        'rel_mse': history.get('rel_mse', [0])[-1],
        'loss_actor': history['loss_actor'][-1],
        'loss_price': history['loss_price'][-1],
        'mean_value_scale': history.get('mean_value_scale', [0])[-1],
    }

    return fig, summary


def plot_baseline_validation(
    policies: Dict[str, callable],
    analytical_fn: callable,
    z_vals: np.ndarray,
    k_bounds: Tuple[float, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learned policies against analytical baseline solution.

    Validates that LR, ER, BR methods converge to the same analytical solution
    in the frictionless (no adjustment cost) case.

    Args:
        policies: Dict mapping method names ('lr', 'er', 'br') to policy networks
                  Each network should accept (k, z) and return k'
        analytical_fn: Function that computes analytical k'(z) given z values
        z_vals: Array of z values to evaluate
        k_bounds: (k_min, k_max) for fixed k evaluation point
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import tensorflow as tf

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Fixed k at midpoint
    k_fixed = (k_bounds[0] + k_bounds[1]) / 2

    # Compute analytical solution
    k_analytical = analytical_fn(z_vals)
    ax.plot(np.log(z_vals), k_analytical, 'k--', linewidth=2.5,
            label='Analytical (frictionless)', zorder=10)

    # Color map for methods
    colors = {'lr': 'blue', 'er': 'orange', 'br': 'green'}
    labels = {'lr': 'LR Method', 'er': 'ER Method', 'br': 'BR Method'}

    # Evaluate each policy
    for method_name, policy_net in policies.items():
        if policy_net is None:
            continue

        # Prepare inputs
        k_input = tf.constant([[k_fixed]] * len(z_vals), dtype=tf.float32)
        z_input = tf.constant(z_vals.reshape(-1, 1), dtype=tf.float32)

        # Get policy output
        k_next = policy_net(k_input, z_input).numpy().flatten()

        color = colors.get(method_name, 'gray')
        label = labels.get(method_name, method_name.upper())
        ax.plot(np.log(z_vals), k_next, color=color, linewidth=2,
                label=label, alpha=0.8)

    ax.set_xlabel('log(z)', fontsize=12)
    ax.set_ylabel("k' (next-period capital)", fontsize=12)
    ax.set_title('Policy Validation: Learned vs Analytical (Frictionless)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
