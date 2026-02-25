# SPECTRA subpackage: re-export model, config, and helpers
from typing import Any, Dict, Optional, Type
import torch.nn as nn

from .spectra_config import SPECTRAConfig, SPECTRATextConfig
from .spectra_model import (
    SPECTRAPreTrainedModel,
    SPECTRATextModel,
    SPECTRAForCausalLM,
    SPECTRAForConditionalGeneration,
    SPECTRAModel,
    SPECTRARouter,
    SPECTRAMoE,
    SPECTRAExoskeletonMoEInjector,
    ExpressionProjector,
)
# Alias: ablation code uses SPECTRABlock with router/expert_module_class signature; SPECTRAMoE is the MoE block.
SPECTRABlock = SPECTRAMoE
from .spectra_utils import (
    find_layers_in_model,
    find_mlp_in_layer,
    is_already_moe,
    copy_mlp_weights_to_expert,
    extract_config_info,
)


def upcycle_model_to_moe(
    model: nn.Module,
    moe_config: Dict[str, Any],
    expert_module_class: Optional[Type] = None,
    layer_start_idx: int = 0,
    layer_end_idx: Optional[int] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Inject SPECTRA exoskeleton into an already-loaded base model (in-place).
    Builds SPECTRATextConfig from model.config + moe_config, then runs injector.inject(model).
    """
    from .spectra_model import SPECTRARouter, SPECTRAExoskeletonMoEInjector

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model must have a .config attribute.")
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    spectra_params = moe_config.get("spectra_params") or moe_config.get("model_config") or moe_config
    if isinstance(spectra_params, dict):
        for k, v in spectra_params.items():
            if k not in config_dict:
                config_dict[k] = v
    text_config = SPECTRATextConfig(**config_dict)
    force_upcycle = kwargs.get("force_upcycle", False)
    global_router = SPECTRARouter(text_config)
    injector = SPECTRAExoskeletonMoEInjector(
        spectra_config=text_config,
        global_router=global_router,
        preserve_shared_experts=True,
        copy_expert_weights=force_upcycle,
    )
    return injector.inject(model)


__all__ = [
    "SPECTRAConfig",
    "SPECTRATextConfig",
    "SPECTRAPreTrainedModel",
    "SPECTRATextModel",
    "SPECTRAForCausalLM",
    "SPECTRAForConditionalGeneration",
    "SPECTRAModel",
    "SPECTRARouter",
    "SPECTRAMoE",
    "SPECTRABlock",
    "SPECTRAExoskeletonMoEInjector",
    "ExpressionProjector",
    "find_layers_in_model",
    "find_mlp_in_layer",
    "is_already_moe",
    "copy_mlp_weights_to_expert",
    "extract_config_info",
    "upcycle_model_to_moe",
]
