"""
plots.py — Publication-quality visualization of simulation results.

Generates four plots as described in the project spec:
  Plot 1: Truthfulness rate vs α (baseline + 3 fixes)
  Plot 2: Mutual information vs α (baseline + 3 fixes)
  Plot 3: Score variance vs α (baseline + 3 fixes)
  Plot 4: Equilibrium robustness — E[honest] vs E[LLM] vs α, with α* marked

All plots use consistent styling: same method-color mapping, clear axis labels,
legends, and α* marked where applicable. Output is saved as PNG files.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import Optional
import os

# Consistent color and style mapping for each method
METHOD_STYLES = {
    'baseline': {'color': '#2c7bb6', 'linestyle': '-',  'marker': 'o', 'label': 'Baseline'},
    'fix_a_k3': {'color': '#d7191c', 'linestyle': '--', 'marker': 's', 'label': 'Fix A (k=3)'},
    'fix_a_k5': {'color': '#fdae61', 'linestyle': '--', 'marker': '^', 'label': 'Fix A (k=5)'},
    'fix_b':    {'color': '#1a9641', 'linestyle': '-.',  'marker': 'D', 'label': 'Fix B (penalty)'},
    'fix_c':    {'color': '#7b2d8b', 'linestyle': ':',  'marker': 'v', 'label': 'Fix C (variance)'},
}

FIGURE_SIZE = (8, 5)
DPI = 150
FONT_SIZE = 12


def _setup_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent formatting to an axes object."""
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE + 1, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.02, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _add_alpha_star(ax, df: pd.DataFrame, y_min: float, y_max: float) -> None:
    """Add a vertical dashed line at α* (equilibrium collapse point) if detected."""
    from experiments import find_alpha_star
    alpha_star = find_alpha_star(df, 'baseline')

    if alpha_star is not None:
        ax.axvline(x=alpha_star, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, zorder=5)
        # Place label above the midpoint
        mid_y = y_min + 0.7 * (y_max - y_min)
        ax.text(alpha_star + 0.02, mid_y, f'α*={alpha_star:.2f}',
                fontsize=FONT_SIZE - 1, color='black', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def plot_truthfulness_rate(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 1: Truthfulness rate vs α for all methods.

    Shows how honest reporting degrades as LLM workers dominate. Under all-honest
    workers (α=0), truthfulness rate should be 1.0. As α increases, LLM workers'
    reports sometimes match their true signal by chance (when consensus = signal),
    but this drops below 1.0 since their reports are driven by consensus, not signal.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    save_path : str, optional
        File path to save the figure. If None, does not save.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    methods = df['scoring_method'].unique()

    for method in methods:
        if method not in METHOD_STYLES:
            continue
        style = METHOD_STYLES[method]
        sub = df[df['scoring_method'] == method].sort_values('alpha')
        ax.plot(sub['alpha'], sub['truthfulness_rate'],
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=5, linewidth=2,
                label=style['label'])

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylim(-0.05, 1.1)

    _add_alpha_star(ax, df, -0.05, 1.1)
    _setup_ax(ax, 'α (fraction of LLM workers)', 'Truthfulness rate',
              'Plot 1: Truthfulness Rate vs. α')
    ax.legend(fontsize=FONT_SIZE - 1, loc='lower left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_mutual_information(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2: Mutual information I(reports; θ) vs α for all methods.

    This is the most important diagnostic plot. As α increases, the reports
    become more correlated with each other (high agreement) but less correlated
    with θ (low information). MI should drop toward 0 as α → 1.

    At α=0, MI should be close to H(majority vote | θ) — the information
    content of the optimal majority vote estimator.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    save_path : str, optional
        File path to save the figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    methods = df['scoring_method'].unique()

    # Theoretical maximum MI (for reference)
    # At α=0, all workers honest with q=0.7, N=200 → majority vote ≈ correct always
    # MI ≈ ln(2) ≈ 0.693 nats (1 bit) when fully informative
    mi_max = np.log(2)
    ax.axhline(y=mi_max, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label=f'Max MI = ln(2) = {mi_max:.3f} nats')

    for method in methods:
        if method not in METHOD_STYLES:
            continue
        style = METHOD_STYLES[method]
        sub = df[df['scoring_method'] == method].sort_values('alpha')
        ax.plot(sub['alpha'], sub['mutual_information'],
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=5, linewidth=2,
                label=style['label'])

    ax.set_ylim(-0.01, mi_max * 1.15)

    _add_alpha_star(ax, df, -0.01, mi_max * 1.15)
    _setup_ax(ax, 'α (fraction of LLM workers)', 'Mutual information I(reports; θ) [nats]',
              'Plot 2: Mutual Information vs. α')
    ax.legend(fontsize=FONT_SIZE - 1, loc='upper right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_score_variance(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3: Score variance vs α for all methods.

    LLM workers produce highly correlated reports, which reduces variance of
    payments (everyone gets the same score when everyone reports the same thing).
    Paradoxically, low variance here is bad — it means the scoring rule is not
    differentiating between workers based on signal quality.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    save_path : str, optional
        File path to save the figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    methods = df['scoring_method'].unique()

    for method in methods:
        if method not in METHOD_STYLES:
            continue
        style = METHOD_STYLES[method]
        sub = df[df['scoring_method'] == method].sort_values('alpha')
        ax.plot(sub['alpha'], sub['score_variance'],
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=5, linewidth=2,
                label=style['label'])

    _add_alpha_star(ax, df,
                    df['score_variance'].min() * 0.9,
                    df['score_variance'].max() * 1.1)
    _setup_ax(ax, 'α (fraction of LLM workers)', 'Variance of log-score payments',
              'Plot 3: Score Variance vs. α')
    ax.legend(fontsize=FONT_SIZE - 1, loc='upper right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_equilibrium_robustness(df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 4: E[honest payment] vs E[LLM payment] vs α, with α* clearly marked.

    This is the equilibrium diagnostic plot. When honest payment > LLM payment,
    honest reporting is the rational strategy. When they cross, the truthful
    Nash equilibrium collapses — this crossing point is α*.

    Shows only the baseline method here (the fixes comparison is in plots 1-3).
    Also shows the theoretical gap from scoring.py for validation.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    save_path : str, optional
        File path to save the figure

    Returns
    -------
    plt.Figure
    """
    from experiments import find_alpha_star

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Baseline only for this plot (cleaner, focused on equilibrium)
    sub = df[df['scoring_method'] == 'baseline'].sort_values('alpha')

    ax.plot(sub['alpha'], sub['honest_payment'],
            color='#2c7bb6', linestyle='-', marker='o', markersize=5,
            linewidth=2, label='E[payment | honest worker]')

    # Only plot LLM payment where LLM workers exist (α > 0)
    llm_data = sub[sub['alpha'] > 0]
    if not llm_data.empty:
        ax.plot(llm_data['alpha'], llm_data['llm_payment'],
                color='#d7191c', linestyle='--', marker='s', markersize=5,
                linewidth=2, label='E[payment | LLM worker]')

    # Mark α*
    alpha_star = find_alpha_star(df, 'baseline')
    if alpha_star is not None:
        y_range = sub['honest_payment'].max() - sub['honest_payment'].min()
        y_min = sub[['honest_payment', 'llm_payment']].min().min()
        y_max = sub[['honest_payment', 'llm_payment']].max().max()
        y_span = y_max - y_min if y_max > y_min else 1.0

        ax.axvline(x=alpha_star, color='black', linestyle='--', linewidth=2,
                   alpha=0.8, label=f'α* = {alpha_star:.2f} (equilibrium collapse)')
        ax.text(alpha_star + 0.02, y_min + 0.2 * y_span,
                f'α* = {alpha_star:.2f}\nTruthful equilibrium\ncollapses here',
                fontsize=FONT_SIZE - 2, color='black', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='black', alpha=0.8))

    # Add shaded region where honest > LLM (mechanism working)
    honest_arr = sub['honest_payment'].values
    llm_arr = sub['llm_payment'].values
    alpha_arr = sub['alpha'].values

    # Fill region where honest > llm (only where llm exists)
    valid = ~np.isnan(llm_arr)
    if valid.any():
        ax.fill_between(alpha_arr[valid],
                         honest_arr[valid], llm_arr[valid],
                         where=honest_arr[valid] >= llm_arr[valid],
                         alpha=0.1, color='#2c7bb6', label='Honest advantage region')
        ax.fill_between(alpha_arr[valid],
                         honest_arr[valid], llm_arr[valid],
                         where=honest_arr[valid] < llm_arr[valid],
                         alpha=0.1, color='#d7191c', label='LLM advantage region')

    _setup_ax(ax, 'α (fraction of LLM workers)',
              'Expected log-score payment',
              'Plot 4: Equilibrium Robustness — Honest vs. LLM Payment')
    ax.legend(fontsize=FONT_SIZE - 1, loc='best')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def generate_all_plots(df: pd.DataFrame, output_dir: str = '.') -> None:
    """
    Generate and save all four plots to the specified output directory.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_all_experiments
    output_dir : str
        Directory to save plot PNG files
    """
    os.makedirs(output_dir, exist_ok=True)

    plots = [
        ('plot1_truthfulness.png', plot_truthfulness_rate),
        ('plot2_mutual_info.png', plot_mutual_information),
        ('plot3_score_variance.png', plot_score_variance),
        ('plot4_equilibrium.png', plot_equilibrium_robustness),
    ]

    for filename, plot_fn in plots:
        path = os.path.join(output_dir, filename)
        fig = plot_fn(df, save_path=path)
        plt.close(fig)

    print(f"\nAll 4 plots saved to: {os.path.abspath(output_dir)}")
