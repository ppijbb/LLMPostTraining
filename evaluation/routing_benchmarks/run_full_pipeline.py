#!/usr/bin/env python3
"""
Full Pipeline Runner for Seqorth 7-Day Evaluation

Automatically runs all evaluation days from D-Day to D+6.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrate the full 7-day evaluation pipeline."""
    
    def __init__(
        self,
        config_path: str,
        base_dir: str = "./evaluation_results",
        continue_on_error: bool = False
    ):
        """
        Initialize pipeline runner.
        
        Args:
            config_path: Path to evaluation_config.yaml
            base_dir: Base directory for results
            continue_on_error: Whether to continue if a step fails
        """
        self.config_path = Path(config_path)
        self.base_dir = Path(base_dir)
        self.continue_on_error = continue_on_error
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Pipeline steps
        self.steps = [
            {
                "name": "day0",
                "description": "Sanity Check (Training Dynamics & PPL)",
                "script": "day0_sanity_check.py",
                "critical": True  # Pipeline stops if this fails
            },
            {
                "name": "day1_2",
                "description": "Standard Benchmarks (lm-eval-harness)",
                "script": "day1_2_standard_benchmarks.py",
                "critical": False
            },
            {
                "name": "day3_4",
                "description": "Expert Analysis (Specialization)",
                "script": "day3_4_expert_analysis.py",
                "critical": False
            },
            {
                "name": "day5",
                "description": "Efficiency & Ablation (vLLM)",
                "script": "day5_efficiency_ablation.py",
                "critical": False
            },
            {
                "name": "day6",
                "description": "Final Comparison Tables",
                "script": "day6_comparison_table.py",
                "critical": False
            }
        ]
        
        self.results = {}
        self.log_file = self.base_dir / "full_pipeline_log.txt"
    
    def run_step(self, step: Dict[str, Any]) -> bool:
        """
        Run a single pipeline step.
        
        Args:
            step: Step configuration dictionary
        
        Returns:
            True if successful, False otherwise
        """
        step_name = step["name"]
        script_name = step["script"]
        
        logger.info("=" * 80)
        logger.info(f"Running {step_name}: {step['description']}")
        logger.info("=" * 80)
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--config", str(self.config_path),
            "--output_dir", str(self.base_dir / step_name)
        ]
        
        # Add checkpoint if specified
        if "checkpoint_path" in self.config["model"]:
            cmd.extend(["--checkpoint", self.config["model"]["checkpoint_path"]])
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run step
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✓ {step_name} completed successfully in {duration:.1f}s")
            
            self.results[step_name] = {
                "status": "success",
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Log output
            if result.stdout:
                logger.debug(f"{step_name} stdout:\n{result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"✗ {step_name} failed after {duration:.1f}s")
            logger.error(f"Return code: {e.returncode}")
            
            if e.stdout:
                logger.error(f"stdout:\n{e.stdout}")
            if e.stderr:
                logger.error(f"stderr:\n{e.stderr}")
            
            self.results[step_name] = {
                "status": "failed",
                "duration": duration,
                "error": str(e),
                "return_code": e.returncode
            }
            
            return False
    
    def run(
        self,
        start_from: Optional[str] = None,
        end_at: Optional[str] = None,
        only: Optional[str] = None
    ) -> bool:
        """
        Run the full pipeline or a subset.
        
        Args:
            start_from: Start from this step (inclusive)
            end_at: End at this step (inclusive)
            only: Run only this specific step
        
        Returns:
            True if all steps succeeded, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Seqorth 7-Day Evaluation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Continue on error: {self.continue_on_error}")
        logger.info("")
        
        # Create output directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which steps to run
        steps_to_run = []
        
        if only:
            # Run only specific step
            for step in self.steps:
                if step["name"] == only:
                    steps_to_run = [step]
                    break
            if not steps_to_run:
                logger.error(f"Unknown step: {only}")
                return False
        else:
            # Run range of steps
            in_range = start_from is None
            
            for step in self.steps:
                if start_from and step["name"] == start_from:
                    in_range = True
                
                if in_range:
                    steps_to_run.append(step)
                
                if end_at and step["name"] == end_at:
                    break
        
        logger.info(f"Steps to run: {[s['name'] for s in steps_to_run]}")
        logger.info("")
        
        # Run steps
        all_success = True
        
        for i, step in enumerate(steps_to_run, 1):
            logger.info(f"[{i}/{len(steps_to_run)}] Starting {step['name']}...")
            
            success = self.run_step(step)
            
            if not success:
                all_success = False
                
                if step.get("critical", False):
                    logger.error(f"Critical step {step['name']} failed. Stopping pipeline.")
                    break
                elif not self.continue_on_error:
                    logger.error(f"Step {step['name']} failed. Stopping pipeline.")
                    break
                else:
                    logger.warning(f"Step {step['name']} failed. Continuing...")
            
            logger.info("")
        
        # Generate final report
        self.generate_report()
        
        logger.info("=" * 80)
        if all_success:
            logger.info("✓ Pipeline completed successfully!")
        else:
            logger.info("✗ Pipeline completed with errors")
        logger.info(f"Results saved to: {self.base_dir}")
        logger.info(f"Log file: {self.log_file}")
        logger.info("=" * 80)
        
        return all_success
    
    def generate_report(self):
        """Generate pipeline execution report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Seqorth 7-Day Evaluation Pipeline Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Config: {self.config_path}")
        report_lines.append(f"Output Directory: {self.base_dir}")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        report_lines.append("Step Results:")
        report_lines.append("-" * 80)
        
        for step in self.steps:
            step_name = step["name"]
            if step_name in self.results:
                result = self.results[step_name]
                status_icon = "✓" if result["status"] == "success" else "✗"
                duration = result.get("duration", 0)
                
                report_lines.append(f"{status_icon} {step_name}: {result['status']} ({duration:.1f}s)")
                if "error" in result:
                    report_lines.append(f"    Error: {result['error']}")
            else:
                report_lines.append(f"  {step_name}: not run")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Count successes
        total_run = len(self.results)
        successes = sum(1 for r in self.results.values() if r["status"] == "success")
        
        report_lines.append(f"Summary: {successes}/{total_run} steps succeeded")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(self.log_file, 'w') as f:
            f.write(report_text)
        
        # Also save JSON
        import json
        json_file = self.base_dir / "pipeline_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + report_text + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Seqorth 7-Day Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_full_pipeline.py --config config/evaluation_config.yaml
  
  # Run from day1_2 onwards
  python run_full_pipeline.py --config config/evaluation_config.yaml --start_from day1_2
  
  # Run only day3_4
  python run_full_pipeline.py --config config/evaluation_config.yaml --only day3_4
  
  # Continue on errors
  python run_full_pipeline.py --config config/evaluation_config.yaml --continue_on_error
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation_config.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Base output directory"
    )
    parser.add_argument(
        "--start_from",
        type=str,
        choices=["day0", "day1_2", "day3_4", "day5", "day6"],
        help="Start from this step (inclusive)"
    )
    parser.add_argument(
        "--end_at",
        type=str,
        choices=["day0", "day1_2", "day3_4", "day5", "day6"],
        help="End at this step (inclusive)"
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["day0", "day1_2", "day3_4", "day5", "day6"],
        help="Run only this specific step"
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = PipelineRunner(
        config_path=args.config,
        base_dir=args.output_dir,
        continue_on_error=args.continue_on_error
    )
    
    success = pipeline.run(
        start_from=args.start_from,
        end_at=args.end_at,
        only=args.only
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

