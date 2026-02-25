"""
G3MoE / SPECTRA Models module

Uses top-level modules for backward compatibility. Subpackages: models.spectra, models.moe.
"""
import os
import torch

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
    torch._dynamo.config.suppress_errors = True

from .moe_config import *
from .moe_model import *
from .g3moe_config import G3MoEConfig, G3MoETextConfig
from .g3moe_model import (
    G3MoEPreTrainedModel,
    G3MoETextModel,
    G3MoEForCausalLM,
    G3MoEForConditionalGeneration,
    G3MoEModel,
)
from .spectra_config import SPECTRAConfig, SPECTRATextConfig
from .spectra_model import (
    SPECTRAPreTrainedModel,
    SPECTRATextModel,
    SPECTRAForCausalLM,
    SPECTRAForConditionalGeneration,
    SPECTRAModel,
    SPECTRAExoskeletonMoEInjector,
)
from .g2moe_config import *
from .g2moe_model import *

__all__ = [
    "G3MoEConfig",
    "G3MoETextConfig",
    "G3MoEPreTrainedModel",
    "G3MoETextModel",
    "G3MoEForCausalLM",
    "G3MoEForConditionalGeneration",
    "G3MoEModel",
    "SPECTRAConfig",
    "SPECTRATextConfig",
    "SPECTRAPreTrainedModel",
    "SPECTRATextModel",
    "SPECTRAForCausalLM",
    "SPECTRAForConditionalGeneration",
    "SPECTRAModel",
    "SPECTRAExoskeletonMoEInjector",
]
