#!/usr/bin/env python3
"""
DeepSpeed Custom Optimizer Registry

Responsibilities
- Provide a clean, idempotent way to register custom optimizers with DeepSpeed
- Expose helpers to construct optimizers from config blocks

Notes
- Must run BEFORE any DeepSpeed/Accelerate Trainer initialization that parses DS JSON
- Avoid import-time side effects; call register_custom_optimizers() explicitly at app entry
"""

import os
import sys
from typing import Dict, Any, Optional, Callable
import torch.optim as optim

# Make sure local package imports resolve regardless of CWD
# Expect PYTHONPATH to include repo root (core/ is under repo root)

from custom_optimizers import LionOptimizer, AdaFactorOptimizer, SophiaOptimizer, MuonOptimizer

_REGISTERED = False


def _safe_append_supported_name(ds_config_cls, name: str) -> None:
    """Safely add an optimizer name to DeepSpeed's supported list/set if available."""
    try:
        container = getattr(ds_config_cls, 'DEEPSPEED_OPTIMIZERS', None)
        if container is None:
            return
        # Handle list or set
        if isinstance(container, (list, tuple)):
            if name not in container:
                container.append(name)  # type: ignore[attr-defined]
        elif isinstance(container, set):
            container.add(name)
    except Exception:
        # Be silent; name validation may differ across DS versions
        pass


def register_custom_optimizers() -> None:
    """
    Register custom optimizers with DeepSpeed in a robust and idempotent manner.

    This should be called as early as possible (module top of entry script) so that
    DS JSON validation (optimizer.type) recognizes custom names like 'MuonOptimizer'.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        import deepspeed
        import deepspeed.runtime.config as DeepSpeedConfig
        import importlib

        # Bind classes into a stable namespace DeepSpeed inspects
        if getattr(deepspeed, 'ops', None) is not None and getattr(deepspeed.ops, 'adam', None) is not None:
            deepspeed.ops.adam.LionOptimizer = LionOptimizer
            deepspeed.ops.adam.AdaFactorOptimizer = AdaFactorOptimizer
            deepspeed.ops.adam.SophiaOptimizer = SophiaOptimizer
            deepspeed.ops.adam.MuonOptimizer = MuonOptimizer

        # Ensure name validation passes (varies across DS versions)
        custom_names = ['MuonOptimizer', 'LionOptimizer', 'AdaFactorOptimizer', 'SophiaOptimizer']
        for name in custom_names:
            _safe_append_supported_name(DeepSpeedConfig, name)

        # Also try module-level whitelists seen in some DS versions
        for mod_path, attr in [
            ('deepspeed.runtime.config', 'SUPPORTED_OPTIMIZERS'),
            ('deepspeed.runtime.config_utils', 'SUPPORTED_OPTIMIZERS'),
        ]:
            try:
                m = importlib.import_module(mod_path)
                cont = getattr(m, attr, None)
                if isinstance(cont, list):
                    for name in custom_names:
                        if name not in cont:
                            cont.append(name)
                elif isinstance(cont, set):
                    for name in custom_names:
                        cont.add(name)
            except Exception:
                # Best-effort only
                raise Exception(f"Failed to register custom optimizers: {exc}")

        print("✅ DeepSpeed custom optimizers registered (namespace + all known whitelists)")
        _REGISTERED = True
    except Exception as exc:
        # Don't crash if DS is unavailable at import; user may be on CPU or no-DS path
        raise Exception(f"⚠️ Error DeepSpeed optimizer registration: {exc}")


def get_optimizer_class(optimizer_name: str):
    """Map a variety of names/aliases to the correct optimizer class."""
    name = (optimizer_name or '').strip().lower()
    alias_map = {
        # Built-ins
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
        # Custom (short names)
        'lion': LionOptimizer,
        'adafactor': AdaFactorOptimizer,
        'sophia': SophiaOptimizer,
        'muon': MuonOptimizer,
        # Custom (class-like names used in JSON)
        'lionoptimizer': LionOptimizer,
        'adafactoroptimizer': AdaFactorOptimizer,
        'sophiaoptimizer': SophiaOptimizer,
        'muonoptimizer': MuonOptimizer,
    }
    return alias_map.get(name)


def create_optimizer_from_config(optimizer_config: Dict[str, Any], model_params):
    """
    Construct an optimizer instance from a config block of shape:
    {"type": <name>, "params": {...}}
    Supports both built-in and custom optimizers.
    """
    optimizer_type = optimizer_config.get('type', 'AdamW')
    optimizer_params = optimizer_config.get('params', {})

    optimizer_class = get_optimizer_class(optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer_class(model_params, **optimizer_params)
