# coding=utf-8
"""Shared helpers for SPECTRA model upcycling and layer discovery."""
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


def find_layers_in_model(model: nn.Module) -> Optional[List[nn.Module]]:
    """Return the list of decoder layers (e.g. model.model.layers or model.language_model.layers)."""
    language_model = model
    if hasattr(model, "language_model"):
        language_model = model.language_model
    elif hasattr(model, "model"):
        language_model = model.model
    if hasattr(language_model, "layers"):
        return language_model.layers
    if hasattr(language_model, "h"):
        return language_model.h
    if hasattr(language_model, "transformer") and hasattr(language_model.transformer, "layers"):
        return language_model.transformer.layers
    if hasattr(language_model, "block"):
        return language_model.block
    if hasattr(language_model, "decoder_layers"):
        return language_model.decoder_layers
    return None


def find_mlp_in_layer(decoder_layer: nn.Module) -> Optional[nn.Module]:
    """Return the MLP / feed-forward module in a decoder layer."""
    for attr in ("mlp", "feed_forward", "ffn", "ffw"):
        if hasattr(decoder_layer, attr):
            return getattr(decoder_layer, attr)
    return None


def is_already_moe(decoder_layer: nn.Module) -> bool:
    """Return True if the layer already has an MoE block."""
    moe = getattr(decoder_layer, "moe", None) or getattr(decoder_layer, "feed_forward", None)
    if moe is None:
        return False
    return hasattr(moe, "experts") or hasattr(moe, "router") or hasattr(moe, "gate")


def copy_mlp_weights_to_expert(source_mlp: nn.Module, expert: nn.Module) -> None:
    """Copy gate_proj, up_proj, down_proj from source_mlp to expert (in place)."""
    with torch.no_grad():
        for attr in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(source_mlp, attr) and hasattr(expert, attr):
                src = getattr(source_mlp, attr)
                dst = getattr(expert, attr)
                if hasattr(src, "weight") and hasattr(dst, "weight") and src.weight.shape == dst.weight.shape:
                    dst.weight.copy_(src.weight)
                if hasattr(src, "bias") and hasattr(dst, "bias") and src.bias is not None and dst.bias is not None:
                    if src.bias.shape == dst.bias.shape:
                        dst.bias.copy_(src.bias)


def extract_config_info(moe_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract flat config dict for upcycle helpers from a nested moe_config."""
    if not isinstance(moe_config, dict):
        return {}
    # Top-level keys
    cfg = dict(moe_config)
    # Nested spectra_params / model_config
    for key in ("spectra_params", "model_config", "moe_params"):
        nested = moe_config.get(key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in cfg:
                    cfg[k] = v
    # Normalize common names
    defaults = {
        "hidden_size": cfg.get("hidden_size", 3072),
        "intermediate_size": cfg.get("intermediate_size", 8192),
        "num_experts": cfg.get("num_experts", 8),
        "num_experts_per_tok": cfg.get("num_experts_per_tok", 2),
        "n_shared_experts": cfg.get("n_shared_experts", 1),
        "first_k_dense_replace": cfg.get("first_k_dense_replace", 0),
        "load_balance_loss_coef": cfg.get("load_balance_loss_coef", 0.01),
        "router_jitter_noise": cfg.get("router_jitter_noise", 0.0),
        "input_jitter_noise": cfg.get("input_jitter_noise", 0.0),
        "freeze_shared_experts": cfg.get("freeze_shared_experts", False),
        "hidden_activation": cfg.get("hidden_activation", "silu"),
        "router_dim": cfg.get("router_dim", 128),
        "balancing_strength": cfg.get("balancing_strength", 0.01),
        "ema_alpha": cfg.get("ema_alpha", 0.99),
    }
    for k, v in defaults.items():
        if k not in cfg or cfg[k] is None:
            cfg[k] = v
    return cfg
