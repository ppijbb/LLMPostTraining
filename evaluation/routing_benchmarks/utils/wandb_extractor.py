"""WandB log extraction utilities for training dynamics analysis."""

import os
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import json

logger = logging.getLogger(__name__)


def extract_training_curves(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract training curves from WandB run.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        metrics: List of metric names to extract (None for all)
        api_key: WandB API key (optional, uses env var if not provided)
    
    Returns:
        Dictionary mapping metric names to DataFrames with columns: [step, value]
    """
    logger.info(f"Extracting training curves from run {run_id}")
    
    # Initialize API
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    
    api = wandb.Api()
    
    # Construct run path
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(run_path)
    except Exception as e:
        logger.error(f"Failed to fetch run {run_path}: {e}")
        raise
    
    # Get history
    history = run.history()
    
    if history.empty:
        logger.warning("No history data found in run")
        return {}
    
    # Extract specified metrics or all
    if metrics is None:
        # Default metrics for SPECTRA
        metrics = [
            "moe/avg_expert_cv",
            "moe/avg_maxvio",
            "train/loss",
            "moe/avg_pairwise_expert_similarity",
            "moe/avg_gram_orthogonality",
            "moe/avg_routing_entropy"
        ]
    
    curves = {}
    for metric in metrics:
        if metric in history.columns:
            df = history[["_step", metric]].dropna()
            df.columns = ["step", "value"]
            curves[metric] = df
            logger.info(f"Extracted {metric}: {len(df)} data points")
        else:
            logger.warning(f"Metric {metric} not found in run history")
    
    return curves


def download_wandb_artifacts(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    artifact_name: Optional[str] = None,
    download_dir: str = "./wandb_artifacts",
    api_key: Optional[str] = None
) -> Path:
    """
    Download artifacts from WandB run (e.g., heatmaps, checkpoints).
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        artifact_name: Specific artifact to download (None for all)
        download_dir: Directory to download artifacts
        api_key: WandB API key (optional)
    
    Returns:
        Path to downloaded artifacts directory
    """
    logger.info(f"Downloading artifacts from run {run_id}")
    
    # Initialize API
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    
    api = wandb.Api()
    
    # Construct run path
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(run_path)
    except Exception as e:
        logger.error(f"Failed to fetch run {run_path}: {e}")
        raise
    
    # Create download directory
    download_path = Path(download_dir) / run_id
    download_path.mkdir(parents=True, exist_ok=True)
    
    # Download artifacts
    artifacts = run.logged_artifacts()
    
    if not artifacts:
        logger.warning("No artifacts found in run")
        return download_path
    
    for artifact in artifacts:
        if artifact_name and artifact.name != artifact_name:
            continue
        
        logger.info(f"Downloading artifact: {artifact.name}")
        artifact_dir = download_path / artifact.name
        artifact.download(root=str(artifact_dir))
    
    logger.info(f"Artifacts downloaded to {download_path}")
    return download_path


def parse_run_metadata(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract run metadata and configuration.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
    
    Returns:
        Dictionary with run metadata
    """
    logger.info(f"Parsing metadata from run {run_id}")
    
    # Initialize API
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    
    api = wandb.Api()
    
    # Construct run path
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"
    
    try:
        run = api.run(run_path)
    except Exception as e:
        logger.error(f"Failed to fetch run {run_path}: {e}")
        raise
    
    # Extract metadata
    metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "runtime": run.summary.get("_runtime", 0),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "tags": run.tags,
        "notes": run.notes,
        "url": run.url,
    }
    
    logger.info(f"Extracted metadata for run {run.name}")
    return metadata


def extract_cv_curve(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract Coefficient of Variation (CV) curve from training.
    
    This is a key metric for SPECTRA showing load balancing.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
    
    Returns:
        DataFrame with columns: [step, cv]
    """
    curves = extract_training_curves(
        run_id=run_id,
        project=project,
        entity=entity,
        metrics=["moe/avg_expert_cv"],
        api_key=api_key
    )
    
    if "moe/avg_expert_cv" in curves:
        df = curves["moe/avg_expert_cv"]
        df.columns = ["step", "cv"]
        return df
    else:
        logger.warning("CV metric not found in run")
        return pd.DataFrame(columns=["step", "cv"])


def extract_maxvio_curve(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract MaxVio (maximum violation) curve from training.
    
    This shows constraint satisfaction in Sinkhorn routing.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
    
    Returns:
        DataFrame with columns: [step, maxvio]
    """
    curves = extract_training_curves(
        run_id=run_id,
        project=project,
        entity=entity,
        metrics=["moe/avg_maxvio"],
        api_key=api_key
    )
    
    if "moe/avg_maxvio" in curves:
        df = curves["moe/avg_maxvio"]
        df.columns = ["step", "maxvio"]
        return df
    else:
        logger.warning("MaxVio metric not found in run")
        return pd.DataFrame(columns=["step", "maxvio"])


def extract_loss_curve(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract loss curve from training.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
    
    Returns:
        DataFrame with columns: [step, loss]
    """
    curves = extract_training_curves(
        run_id=run_id,
        project=project,
        entity=entity,
        metrics=["train/loss"],
        api_key=api_key
    )
    
    if "train/loss" in curves:
        df = curves["train/loss"]
        df.columns = ["step", "loss"]
        return df
    else:
        logger.warning("Loss metric not found in run")
        return pd.DataFrame(columns=["step", "loss"])


def extract_orthogonality_curve(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract Gram matrix orthogonality curve from training.
    
    This shows OSR (Orthogonal Sinkhorn Routing) effectiveness.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
    
    Returns:
        DataFrame with columns: [step, orthogonality]
    """
    curves = extract_training_curves(
        run_id=run_id,
        project=project,
        entity=entity,
        metrics=["moe/avg_gram_orthogonality"],
        api_key=api_key
    )
    
    if "moe/avg_gram_orthogonality" in curves:
        df = curves["moe/avg_gram_orthogonality"]
        df.columns = ["step", "orthogonality"]
        return df
    else:
        logger.warning("Orthogonality metric not found in run")
        return pd.DataFrame(columns=["step", "orthogonality"])


def find_explicit_bias_start_step(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    api_key: Optional[str] = None,
    threshold: float = 0.1
) -> Optional[int]:
    """
    Find the step where Explicit Bias (DeepSeek-V3 style) was applied.
    
    This is detected by a sudden drop in CV.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/team name (optional)
        api_key: WandB API key (optional)
        threshold: CV drop threshold to detect bias start
    
    Returns:
        Step number where bias was applied, or None if not detected
    """
    cv_df = extract_cv_curve(run_id, project, entity, api_key)
    
    if cv_df.empty:
        return None
    
    # Look for sudden drops in CV
    cv_values = cv_df["cv"].values
    steps = cv_df["step"].values
    
    # Compute rolling window derivative
    window = 10
    if len(cv_values) < window * 2:
        return None
    
    derivatives = np.diff(cv_values)
    
    # Find largest negative derivative (biggest drop)
    for i in range(window, len(derivatives) - window):
        if derivatives[i] < -threshold:
            # Check if this is sustained
            if np.mean(cv_values[i+1:i+window]) < cv_values[i] - threshold:
                logger.info(f"Detected bias application at step {steps[i]}")
                return int(steps[i])
    
    logger.warning("Could not detect bias application point")
    return None


def export_curves_to_csv(
    curves: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path]
) -> None:
    """
    Export extracted curves to CSV files.
    
    Args:
        curves: Dictionary of metric DataFrames
        output_dir: Output directory for CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for metric_name, df in curves.items():
        # Sanitize metric name for filename
        safe_name = metric_name.replace("/", "_").replace(":", "_")
        csv_path = output_path / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported {metric_name} to {csv_path}")


def load_curves_from_csv(
    input_dir: Union[str, Path],
    metric_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load curves from CSV files.
    
    Args:
        input_dir: Directory containing CSV files
        metric_name: Specific metric to load (None for all)
    
    Returns:
        Dictionary of metric DataFrames
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return {}
    
    curves = {}
    
    if metric_name:
        # Load specific metric
        safe_name = metric_name.replace("/", "_").replace(":", "_")
        csv_path = input_path / f"{safe_name}.csv"
        if csv_path.exists():
            curves[metric_name] = pd.read_csv(csv_path)
        else:
            logger.warning(f"CSV file not found for {metric_name}")
    else:
        # Load all CSV files
        for csv_file in input_path.glob("*.csv"):
            metric_name = csv_file.stem.replace("_", "/")
            curves[metric_name] = pd.read_csv(csv_file)
    
    logger.info(f"Loaded {len(curves)} curves from {input_path}")
    return curves

