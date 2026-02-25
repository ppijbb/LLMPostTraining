"""Visualization utilities for SPECTRA evaluation results."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_cv_curve(
    cv_data: pd.DataFrame,
    output_file: Optional[Union[str, Path]] = None,
    explicit_bias_step: Optional[int] = None,
    title: str = "Coefficient of Variation (CV) During Training",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot CV curve showing load balancing over training.
    
    Args:
        cv_data: DataFrame with columns [step, cv]
        output_file: Optional file to save figure
        explicit_bias_step: Step where explicit bias was applied
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot CV curve
    ax.plot(cv_data["step"], cv_data["cv"], linewidth=2, label="Expert CV")
    
    # Add horizontal lines for reference
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label="Good (CV < 0.1)")
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label="Acceptable (CV < 0.2)")
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label="Poor (CV > 0.3)")
    
    # Mark explicit bias application point
    if explicit_bias_step is not None:
        ax.axvline(x=explicit_bias_step, color='purple', linestyle=':', linewidth=2,
                   label=f"Explicit Bias Applied (Step {explicit_bias_step})")
        ax.annotate(
            'DeepSeek-V3\nExplicit Bias',
            xy=(explicit_bias_step, cv_data["cv"].max()),
            xytext=(explicit_bias_step + len(cv_data) * 0.05, cv_data["cv"].max() * 0.9),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2),
            fontsize=10,
            color='purple'
        )
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Coefficient of Variation (CV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved CV curve to {output_file}")
    
    return fig


def plot_maxvio_curve(
    maxvio_data: pd.DataFrame,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "MaxVio (Constraint Violation) During Training",
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot MaxVio curve showing constraint satisfaction.
    
    Args:
        maxvio_data: DataFrame with columns [step, maxvio]
        output_file: Optional file to save figure
        title: Plot title
        threshold: Acceptable MaxVio threshold
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot MaxVio curve
    ax.plot(maxvio_data["step"], maxvio_data["maxvio"], linewidth=2, label="MaxVio", color='darkblue')
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f"Threshold (MaxVio < {threshold})")
    
    # Fill area above threshold
    ax.fill_between(
        maxvio_data["step"],
        maxvio_data["maxvio"],
        threshold,
        where=maxvio_data["maxvio"] > threshold,
        alpha=0.3,
        color='red',
        label='Violation Region'
    )
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("MaxVio", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visibility
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved MaxVio curve to {output_file}")
    
    return fig


def plot_expert_heatmap(
    expert_usage: Dict[str, np.ndarray],
    domains: List[str],
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Expert Usage Across Domains",
    top_k: int = 10,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot heatmap of expert usage across different domains.
    
    Args:
        expert_usage: Dict mapping domain names to expert usage arrays [num_experts]
        domains: List of domain names (in order)
        output_file: Optional file to save figure
        title: Plot title
        top_k: Number of top experts to show per domain
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Prepare data for heatmap
    # For each domain, get top-k experts
    heatmap_data = []
    expert_ids = set()
    
    for domain in domains:
        usage = expert_usage[domain]
        top_experts = np.argsort(usage)[-top_k:][::-1]  # Top-k in descending order
        expert_ids.update(top_experts)
    
    expert_ids = sorted(expert_ids)
    
    # Create matrix: rows = experts, columns = domains
    matrix = np.zeros((len(expert_ids), len(domains)))
    
    for j, domain in enumerate(domains):
        usage = expert_usage[domain]
        # Normalize to percentage
        usage_pct = usage / usage.sum() * 100 if usage.sum() > 0 else usage
        for i, expert_id in enumerate(expert_ids):
            matrix[i, j] = usage_pct[expert_id]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(expert_ids)))
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_yticklabels([f"Expert {e}" for e in expert_ids], fontsize=9)
    
    # Rotate domain labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Usage Percentage (%)", rotation=-90, va="bottom", fontsize=11)
    
    # Add text annotations
    for i in range(len(expert_ids)):
        for j in range(len(domains)):
            if matrix[i, j] > 1.0:  # Only show if > 1%
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha="center", va="center", color="black" if matrix[i, j] < 50 else "white",
                             fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Expert ID", fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved expert heatmap to {output_file}")
    
    return fig


def plot_trajectory_consistency(
    spectra_consistency: np.ndarray,
    baseline_consistency: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    output_file: Optional[Union[str, Path]] = None,
    title: str = "GRU Trajectory Consistency",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot trajectory consistency comparison.
    
    Args:
        spectra_consistency: Array of consistency scores for SPECTRA
        baseline_consistency: Optional array for baseline model
        baseline_name: Name of baseline model
        output_file: Optional file to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot SPECTRA
    ax.hist(spectra_consistency, bins=50, alpha=0.7, label="SPECTRA", color='blue', edgecolor='black')
    
    # Plot baseline if provided
    if baseline_consistency is not None:
        ax.hist(baseline_consistency, bins=50, alpha=0.7, label=baseline_name, color='orange', edgecolor='black')
    
    # Add mean lines
    spectra_mean = spectra_consistency.mean()
    ax.axvline(spectra_mean, color='blue', linestyle='--', linewidth=2,
               label=f'SPECTRA Mean: {spectra_mean:.3f}')
    
    if baseline_consistency is not None:
        baseline_mean = baseline_consistency.mean()
        ax.axvline(baseline_mean, color='orange', linestyle='--', linewidth=2,
                   label=f'{baseline_name} Mean: {baseline_mean:.3f}')
    
    ax.set_xlabel("L1 Distance Between Layers (lower = more consistent)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trajectory consistency plot to {output_file}")
    
    return fig


def plot_orthogonality_histogram(
    similarities: np.ndarray,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Pairwise Cosine Similarity of Expert Representations",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of pairwise cosine similarities (OSR validation).
    
    Args:
        similarities: Array of pairwise cosine similarities
        output_file: Optional file to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, bins, patches = ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    
    # Color bars based on value (green near 0, red away from 0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        if abs(c) < 0.1:
            p.set_facecolor('green')
            p.set_alpha(0.8)
        elif abs(c) < 0.3:
            p.set_facecolor('yellow')
            p.set_alpha(0.7)
        else:
            p.set_facecolor('red')
            p.set_alpha(0.6)
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect Orthogonality')
    
    # Add mean and std
    mean_sim = similarities.mean()
    std_sim = similarities.std()
    ax.axvline(mean_sim, color='blue', linestyle=':', linewidth=2,
               label=f'Mean: {mean_sim:.3f} ± {std_sim:.3f}')
    
    # Add shaded region for "good" orthogonality
    ax.axvspan(-0.1, 0.1, alpha=0.2, color='green', label='Good Orthogonality (|sim| < 0.1)')
    
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}\n|sim| < 0.1: {(np.abs(similarities) < 0.1).mean() * 100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved orthogonality histogram to {output_file}")
    
    return fig


def plot_loss_curve(
    loss_data: pd.DataFrame,
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Training Loss (Aux-Loss Free)",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot training loss curve.
    
    Args:
        loss_data: DataFrame with columns [step, loss]
        output_file: Optional file to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot loss
    ax.plot(loss_data["step"], loss_data["loss"], linewidth=2, color='darkred', label="Task Loss Only")
    
    # Add smoothed curve
    if len(loss_data) > 100:
        from scipy.ndimage import uniform_filter1d
        smoothed_loss = uniform_filter1d(loss_data["loss"], size=min(100, len(loss_data) // 10))
        ax.plot(loss_data["step"], smoothed_loss, linewidth=3, color='red',
                alpha=0.7, label="Smoothed", linestyle='--')
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text noting no auxiliary loss
    ax.text(0.02, 0.02, "Note: No Auxiliary Load Balancing Loss Used",
            transform=ax.transAxes, fontsize=10, color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss curve to {output_file}")
    
    return fig


def plot_throughput_comparison(
    spectra_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Inference Throughput Comparison",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot throughput comparison bar chart.
    
    Args:
        spectra_metrics: Dict with keys like "tokens_per_sec", "ttft_ms"
        baseline_metrics: Dict mapping baseline names to their metrics
        output_file: Optional file to save figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    models = ["SPECTRA"] + list(baseline_metrics.keys())
    tokens_per_sec = [spectra_metrics.get("tokens_per_sec", 0)] + \
                     [m.get("tokens_per_sec", 0) for m in baseline_metrics.values()]
    ttft_ms = [spectra_metrics.get("ttft_ms", 0)] + \
              [m.get("ttft_ms", 0) for m in baseline_metrics.values()]
    
    # Plot 1: Tokens/sec (higher is better)
    bars1 = ax1.bar(models, tokens_per_sec, color=['blue'] + ['orange'] * (len(models) - 1))
    ax1.set_ylabel("Tokens / Second", fontsize=11)
    ax1.set_title("Decoding Throughput (Higher is Better)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: TTFT (lower is better)
    bars2 = ax2.bar(models, ttft_ms, color=['blue'] + ['orange'] * (len(models) - 1))
    ax2.set_ylabel("Time to First Token (ms)", fontsize=11)
    ax2.set_title("Prefill Latency (Lower is Better)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Rotate x labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved throughput comparison to {output_file}")
    
    return fig


def create_multi_curve_figure(
    curves: Dict[str, pd.DataFrame],
    output_file: Optional[Union[str, Path]] = None,
    title: str = "Training Dynamics Overview",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create a multi-panel figure with multiple training curves.
    
    Args:
        curves: Dict mapping curve names to DataFrames
        output_file: Optional file to save figure
        title: Overall title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    num_curves = len(curves)
    nrows = (num_curves + 1) // 2
    ncols = 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if num_curves > 1 else [axes]
    
    for idx, (curve_name, df) in enumerate(curves.items()):
        ax = axes[idx]
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], linewidth=2)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(curve_name, fontsize=10)
        ax.set_title(curve_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_curves, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-curve figure to {output_file}")
    
    return fig

