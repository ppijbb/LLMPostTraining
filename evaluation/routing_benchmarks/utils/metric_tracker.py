"""Metric tracking and result persistence for evaluation pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricTracker:
    """
    Track and persist evaluation metrics across pipeline steps.
    
    Supports:
    - Saving/loading results for each day
    - Checkpointing intermediate results
    - Aggregating results across models
    - Version control for reproducibility
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path] = "./evaluation_results",
        use_timestamp: bool = True,
        auto_save: bool = True
    ):
        """
        Initialize metric tracker.
        
        Args:
            base_dir: Base directory for storing results
            use_timestamp: Whether to use timestamped subdirectories
            auto_save: Whether to auto-save after each update
        """
        self.base_dir = Path(base_dir)
        self.auto_save = auto_save
        
        # Create timestamped directory if enabled
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.result_dir = self.base_dir / timestamp
        else:
            self.result_dir = self.base_dir
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal storage
        self.metrics = {}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "pipeline_steps": []
        }
        
        logger.info(f"Initialized MetricTracker at {self.result_dir}")
    
    def add_metric(
        self,
        day: str,
        metric_name: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a metric value.
        
        Args:
            day: Day identifier (e.g., "day0", "day1_2")
            metric_name: Name of the metric
            value: Metric value (can be scalar, list, dict, etc.)
            metadata: Optional metadata for this metric
        """
        if day not in self.metrics:
            self.metrics[day] = {}
        
        self.metrics[day][metric_name] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        logger.debug(f"Added metric {day}/{metric_name}: {value}")
        
        if self.auto_save:
            self.save_day(day)
    
    def get_metric(
        self,
        day: str,
        metric_name: str,
        default: Any = None
    ) -> Any:
        """
        Get a metric value.
        
        Args:
            day: Day identifier
            metric_name: Name of the metric
            default: Default value if metric not found
        
        Returns:
            Metric value or default
        """
        if day in self.metrics and metric_name in self.metrics[day]:
            return self.metrics[day][metric_name]["value"]
        return default
    
    def add_model_result(
        self,
        day: str,
        model_name: str,
        results: Dict[str, Any]
    ) -> None:
        """
        Add results for a specific model.
        
        Args:
            day: Day identifier
            model_name: Name of the model
            results: Dictionary of results
        """
        model_key = f"models/{model_name}"
        self.add_metric(day, model_key, results)
    
    def get_model_result(
        self,
        day: str,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific model.
        
        Args:
            day: Day identifier
            model_name: Name of the model
        
        Returns:
            Dictionary of results or None
        """
        model_key = f"models/{model_name}"
        return self.get_metric(day, model_key)
    
    def save_day(self, day: str) -> Path:
        """
        Save results for a specific day.
        
        Args:
            day: Day identifier
        
        Returns:
            Path to saved file
        """
        day_dir = self.result_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = day_dir / "metrics.json"
        
        # Prepare data for saving
        save_data = {
            "day": day,
            "metrics": self.metrics.get(day, {}),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Saved {day} metrics to {output_file}")
        return output_file
    
    def load_day(self, day: str) -> bool:
        """
        Load results for a specific day.
        
        Args:
            day: Day identifier
        
        Returns:
            True if loaded successfully, False otherwise
        """
        day_dir = self.result_dir / day
        input_file = day_dir / "metrics.json"
        
        if not input_file.exists():
            logger.warning(f"Metrics file not found: {input_file}")
            return False
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            self.metrics[day] = data.get("metrics", {})
            logger.info(f"Loaded {day} metrics from {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {day} metrics: {e}")
            return False
    
    def save_all(self) -> None:
        """Save all metrics to disk."""
        for day in self.metrics.keys():
            self.save_day(day)
        
        # Save metadata
        metadata_file = self.result_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved all metrics and metadata")
    
    def load_all(self) -> None:
        """Load all available metrics from disk."""
        # Load metadata
        metadata_file = self.result_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Load all day directories
        for day_dir in self.result_dir.iterdir():
            if day_dir.is_dir() and day_dir.name.startswith("day"):
                self.load_day(day_dir.name)
        
        logger.info(f"Loaded metrics for {len(self.metrics)} days")
    
    def has_day_results(self, day: str) -> bool:
        """
        Check if results exist for a day.
        
        Args:
            day: Day identifier
        
        Returns:
            True if results exist
        """
        day_dir = self.result_dir / day
        input_file = day_dir / "metrics.json"
        return input_file.exists()
    
    def aggregate_results(
        self,
        days: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate results across days.
        
        Args:
            days: List of days to aggregate (None for all)
        
        Returns:
            Aggregated results dictionary
        """
        if days is None:
            days = list(self.metrics.keys())
        
        aggregated = {}
        
        for day in days:
            if day in self.metrics:
                aggregated[day] = {}
                for metric_name, metric_data in self.metrics[day].items():
                    aggregated[day][metric_name] = metric_data["value"]
        
        return aggregated
    
    def export_summary(
        self,
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Export a summary of all results.
        
        Args:
            output_file: Optional file to save summary
        
        Returns:
            Summary dictionary
        """
        summary = {
            "metadata": self.metadata,
            "results": self.aggregate_results(),
            "generated_at": datetime.now().isoformat()
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Exported summary to {output_path}")
        
        return summary
    
    def mark_step_complete(self, step: str) -> None:
        """
        Mark a pipeline step as complete.
        
        Args:
            step: Step identifier
        """
        self.metadata["pipeline_steps"].append({
            "step": step,
            "completed_at": datetime.now().isoformat()
        })
        
        # Save metadata
        metadata_file = self.result_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def is_step_complete(self, step: str) -> bool:
        """
        Check if a pipeline step is complete.
        
        Args:
            step: Step identifier
        
        Returns:
            True if step is complete
        """
        for step_data in self.metadata["pipeline_steps"]:
            if step_data["step"] == step:
                return True
        return False
    
    def get_comparison_table_data(
        self,
        day: str,
        models: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get data formatted for comparison table generation.
        
        Args:
            day: Day identifier
            models: List of model names
            metrics: List of metric names
        
        Returns:
            Dictionary mapping models to their metric values
        """
        table_data = {}
        
        for model in models:
            model_results = self.get_model_result(day, model)
            if model_results:
                table_data[model] = {}
                for metric in metrics:
                    table_data[model][metric] = model_results.get(metric)
        
        return table_data
    
    def save_checkpoint(self, checkpoint_name: str) -> Path:
        """
        Save a named checkpoint of current state.
        
        Args:
            checkpoint_name: Name for this checkpoint
        
        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = self.result_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.json"
        
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "created_at": datetime.now().isoformat(),
            "metrics": self.metrics,
            "metadata": self.metadata
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Load a named checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load
        
        Returns:
            True if loaded successfully
        """
        checkpoint_dir = self.result_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.metrics = checkpoint_data["metrics"]
            self.metadata = checkpoint_data["metadata"]
            
            logger.info(f"Loaded checkpoint: {checkpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_result_dir(self) -> Path:
        """Get the result directory path."""
        return self.result_dir
    
    def __repr__(self) -> str:
        return f"MetricTracker(dir={self.result_dir}, days={len(self.metrics)})"

