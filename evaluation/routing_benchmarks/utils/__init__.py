"""Utility modules for SPECTRA evaluation pipeline."""

from .checkpoint_loader import load_spectra_checkpoint, load_baseline_model, prepare_model_for_eval
from .wandb_extractor import extract_training_curves, download_wandb_artifacts, parse_run_metadata
from .metric_tracker import MetricTracker
from .visualization import (
    plot_cv_curve,
    plot_maxvio_curve,
    plot_expert_heatmap,
    plot_trajectory_consistency,
    plot_orthogonality_histogram
)
from .latex_formatter import generate_comparison_table, generate_ablation_table, format_metrics

__all__ = [
    'load_spectra_checkpoint',
    'load_baseline_model',
    'prepare_model_for_eval',
    'extract_training_curves',
    'download_wandb_artifacts',
    'parse_run_metadata',
    'MetricTracker',
    'plot_cv_curve',
    'plot_maxvio_curve',
    'plot_expert_heatmap',
    'plot_trajectory_consistency',
    'plot_orthogonality_histogram',
    'generate_comparison_table',
    'generate_ablation_table',
    'format_metrics',
]

