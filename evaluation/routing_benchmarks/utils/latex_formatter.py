"""LaTeX table generation utilities for paper-ready results."""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def format_metrics(
    value: Any,
    format_spec: str = "{:.2f}",
    lower_is_better: bool = False
) -> str:
    """
    Format a metric value for display.
    
    Args:
        value: Metric value
        format_spec: Format specification
        lower_is_better: Whether lower values are better (for highlighting)
    
    Returns:
        Formatted string
    """
    if value is None:
        return "-"
    
    try:
        if isinstance(value, (int, float, np.number)):
            return format_spec.format(value)
        else:
            return str(value)
    except Exception as e:
        logger.warning(f"Failed to format value {value}: {e}")
        return str(value)


def generate_comparison_table(
    model_results: Dict[str, Dict[str, Any]],
    metrics: List[Dict[str, str]],
    output_file: Optional[Union[str, Path]] = None,
    caption: str = "Model comparison on standard benchmarks.",
    label: str = "tab:comparison",
    bold_best: bool = True,
    column_alignment: Optional[str] = None
) -> str:
    """
    Generate LaTeX comparison table.
    
    Args:
        model_results: Dict mapping model names to their results
        metrics: List of metric specs with keys: name, key, format, bold_best, lower_is_better
        output_file: Optional file to save LaTeX code
        caption: Table caption
        label: Table label for referencing
        bold_best: Whether to bold the best value in each column
        column_alignment: Column alignment string (e.g., "lcccc")
    
    Returns:
        LaTeX table string
    """
    # Prepare column alignment
    if column_alignment is None:
        column_alignment = "l" + "c" * len(metrics)
    
    # Start table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{" + caption + r"}")
    latex.append(r"  \label{" + label + r"}")
    latex.append(r"  \begin{tabular}{" + column_alignment + r"}")
    latex.append(r"    \toprule")
    
    # Header row
    header = ["Model"] + [m["name"] for m in metrics]
    latex.append("    " + " & ".join(header) + r" \\")
    latex.append(r"    \midrule")
    
    # Data rows
    for model_name, results in model_results.items():
        row = [model_name]
        
        for metric in metrics:
            metric_key = metric["key"]
            format_spec = metric.get("format", "{:.2f}")
            value = results.get(metric_key)
            
            formatted = format_metrics(value, format_spec)
            
            # Bold if best
            if bold_best and metric.get("bold_best", False) and value is not None:
                lower_is_better = metric.get("lower_is_better", False)
                
                # Collect all values for this metric
                all_values = []
                for other_results in model_results.values():
                    other_value = other_results.get(metric_key)
                    if other_value is not None:
                        all_values.append(other_value)
                
                if all_values:
                    if lower_is_better:
                        best_value = min(all_values)
                    else:
                        best_value = max(all_values)
                    
                    if abs(value - best_value) < 1e-6:  # Handle floating point comparison
                        formatted = r"\textbf{" + formatted + r"}"
            
            row.append(formatted)
        
        latex.append("    " + " & ".join(row) + r" \\")
    
    # End table
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {output_path}")
    
    return latex_str


def generate_ablation_table(
    ablation_results: Dict[str, Dict[str, Any]],
    metrics: List[str],
    output_file: Optional[Union[str, Path]] = None,
    caption: str = "Ablation study results.",
    label: str = "tab:ablation",
    baseline_config: str = "Full"
) -> str:
    """
    Generate LaTeX ablation study table.
    
    Args:
        ablation_results: Dict mapping config names to their results
        metrics: List of metric names to include
        output_file: Optional file to save LaTeX code
        caption: Table caption
        label: Table label
        baseline_config: Name of the baseline (full) configuration
    
    Returns:
        LaTeX table string
    """
    # Determine column alignment
    column_alignment = "l" + "c" * len(metrics) + "c"
    
    # Start table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{" + caption + r"}")
    latex.append(r"  \label{" + label + r"}")
    latex.append(r"  \begin{tabular}{" + column_alignment + r"}")
    latex.append(r"    \toprule")
    
    # Header row
    header = ["Configuration"] + metrics + [r"$\Delta$ PPL"]
    latex.append("    " + " & ".join(header) + r" \\")
    latex.append(r"    \midrule")
    
    # Get baseline perplexity for delta calculation
    baseline_ppl = None
    if baseline_config in ablation_results:
        baseline_ppl = ablation_results[baseline_config].get("perplexity")
    
    # Data rows
    for config_name, results in ablation_results.items():
        row = [config_name]
        
        # Add metric values
        for metric in metrics:
            value = results.get(metric)
            if metric.lower().endswith("ppl") or metric.lower() == "perplexity":
                formatted = format_metrics(value, "{:.2f}")
            else:
                formatted = format_metrics(value, "{:.1f}")
            row.append(formatted)
        
        # Calculate delta PPL
        if baseline_ppl is not None and config_name != baseline_config:
            current_ppl = results.get("perplexity")
            if current_ppl is not None:
                delta = current_ppl - baseline_ppl
                delta_str = f"{delta:+.2f}"  # + or - prefix
                if delta > 0:  # Worse (higher PPL)
                    delta_str = r"\textcolor{red}{" + delta_str + r"}"
                row.append(delta_str)
            else:
                row.append("-")
        else:
            row.append("-")  # Baseline has no delta
        
        # Bold baseline row
        if config_name == baseline_config:
            row = [r"\textbf{" + cell + r"}" if not cell.startswith(r"\text") else cell for cell in row]
        
        latex.append("    " + " & ".join(row) + r" \\")
    
    # End table
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved ablation table to {output_path}")
    
    return latex_str


def generate_parameter_table(
    model_specs: Dict[str, Dict[str, Any]],
    output_file: Optional[Union[str, Path]] = None,
    caption: str = "Model specifications.",
    label: str = "tab:model_specs"
) -> str:
    """
    Generate LaTeX table showing model parameters and specifications.
    
    Args:
        model_specs: Dict mapping model names to their specifications
        output_file: Optional file to save LaTeX code
        caption: Table caption
        label: Table label
    
    Returns:
        LaTeX table string
    """
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{" + caption + r"}")
    latex.append(r"  \label{" + label + r"}")
    latex.append(r"  \begin{tabular}{lccccc}")
    latex.append(r"    \toprule")
    latex.append(r"    Model & Total Params & Active Params & \#Experts & Top-$k$ & Router Type \\")
    latex.append(r"    \midrule")
    
    for model_name, specs in model_specs.items():
        total_params = specs.get("total_params", "-")
        active_params = specs.get("active_params", "-")
        num_experts = specs.get("num_experts", "-")
        top_k = specs.get("top_k", "-")
        router_type = specs.get("router_type", "-")
        
        # Format parameters in billions
        if isinstance(total_params, (int, float)):
            total_params = f"{total_params / 1e9:.1f}B"
        if isinstance(active_params, (int, float)):
            active_params = f"{active_params / 1e9:.1f}B"
        
        row = [model_name, total_params, active_params, str(num_experts), str(top_k), router_type]
        latex.append("    " + " & ".join(row) + r" \\")
    
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved parameter table to {output_path}")
    
    return latex_str


def generate_benchmark_table(
    benchmark_results: Dict[str, Dict[str, float]],
    output_file: Optional[Union[str, Path]] = None,
    caption: str = "Benchmark results across different task categories.",
    label: str = "tab:benchmarks"
) -> str:
    """
    Generate LaTeX table for benchmark results organized by category.
    
    Args:
        benchmark_results: Dict mapping model names to benchmark results
        output_file: Optional file to save LaTeX code
        caption: Table caption
        label: Table label
    
    Returns:
        LaTeX table string
    """
    # Organize benchmarks by category
    categories = {
        "Knowledge": ["mmlu"],
        "Reasoning": ["gsm8k", "arc_challenge"],
        "Commonsense": ["hellaswag", "winogrande"],
        "Coding": ["humaneval", "mbpp"]
    }
    
    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{" + caption + r"}")
    latex.append(r"  \label{" + label + r"}")
    
    # Count total benchmarks
    all_benchmarks = [b for benches in categories.values() for b in benches]
    column_alignment = "l" + "c" * len(all_benchmarks)
    
    latex.append(r"  \begin{tabular}{" + column_alignment + r"}")
    latex.append(r"    \toprule")
    
    # Multi-row header
    header1 = ["\\multirow{2}{*}{Model}"]
    header2 = []
    
    for category, benchmarks in categories.items():
        header1.append(f"\\multicolumn{{{len(benchmarks)}}}{{c}}{{{category}}}")
        header2.extend([b.upper() for b in benchmarks])
    
    latex.append("    " + " & ".join(header1) + r" \\")
    latex.append(r"    \cmidrule(lr){2-" + str(len(all_benchmarks) + 1) + r"}")
    latex.append("    " + " & " + " & ".join(header2) + r" \\")
    latex.append(r"    \midrule")
    
    # Data rows
    for model_name, results in benchmark_results.items():
        row = [model_name]
        
        for benchmarks in categories.values():
            for benchmark in benchmarks:
                value = results.get(benchmark)
                formatted = format_metrics(value, "{:.1f}")
                
                # Bold best in column
                all_values = [r.get(benchmark) for r in benchmark_results.values() if r.get(benchmark) is not None]
                if all_values and value is not None and value == max(all_values):
                    formatted = r"\textbf{" + formatted + r"}"
                
                row.append(formatted)
        
        latex.append("    " + " & ".join(row) + r" \\")
    
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table*}")
    
    latex_str = "\n".join(latex)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved benchmark table to {output_path}")
    
    return latex_str


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in text.
    
    Args:
        text: Input text
    
    Returns:
        LaTeX-safe text
    """
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text


def format_number_with_uncertainty(
    value: float,
    uncertainty: float,
    format_spec: str = "{:.2f}"
) -> str:
    """
    Format a number with uncertainty (e.g., for mean ± std).
    
    Args:
        value: Mean value
        uncertainty: Uncertainty/std
        format_spec: Format specification
    
    Returns:
        Formatted string like "12.34 $\\pm$ 0.56"
    """
    value_str = format_spec.format(value)
    uncertainty_str = format_spec.format(uncertainty)
    return f"{value_str} $\\pm$ {uncertainty_str}"

