# Pytest conftest: ensure repo root is on PYTHONPATH when running tests
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# HybridCache Mock Patch for environments with legacy/incompatible transformers version
import torch
try:
    import transformers.cache_utils
    if not hasattr(transformers.cache_utils, "HybridCache"):
        class DummyHybridCache(transformers.cache_utils.Cache):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.key_cache = []
                self.value_cache = []
                self.seen_tokens = 0
                
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                if len(self.key_cache) <= layer_idx:
                    self.key_cache.append(key_states)
                    self.value_cache.append(value_states)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
                
            def get_seq_length(self, layer_idx=0):
                if len(self.key_cache) <= layer_idx:
                    return 0
                return self.key_cache[layer_idx].shape[-2]
                
            def get_max_length(self):
                return None
                
        transformers.cache_utils.HybridCache = DummyHybridCache
        sys.modules["transformers.cache_utils"] = transformers.cache_utils
except ImportError:
    pass

# modeling_utils patches
try:
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils, "restore_default_dtype"):
        def restore_default_dtype(func):
            return func
        transformers.modeling_utils.restore_default_dtype = restore_default_dtype
        
    if not hasattr(transformers.modeling_utils, "SpecificPreTrainedModelType"):
        class SpecificPreTrainedModelType:
            pass
        transformers.modeling_utils.SpecificPreTrainedModelType = SpecificPreTrainedModelType
        
    if not hasattr(transformers.modeling_utils, "PretrainedConfig"):
        import transformers
        transformers.modeling_utils.PretrainedConfig = transformers.PreTrainedConfig
except ImportError:
    pass

# Deepspeed mock patch
try:
    import types
    import importlib.machinery
    deepspeed_mod = types.ModuleType("deepspeed")
    deepspeed_mod.__spec__ = importlib.machinery.ModuleSpec(
        name="deepspeed",
        loader=None,
        origin=None,
        is_package=True
    )
    class DummyGatheredParameters:
        def __init__(self, params, modifier_rank=None, enabled=True):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class DummyZero:
        GatheredParameters = DummyGatheredParameters

    deepspeed_mod.zero = DummyZero()
    sys.modules["deepspeed"] = deepspeed_mod
except Exception:
    pass

# Einops mock patch
try:
    class DummyEinops:
        @staticmethod
        def rearrange(tensor, pattern, **axes_lengths):
            return tensor
    sys.modules["einops"] = DummyEinops
except Exception:
    pass

# Flash_attn mock patch
try:
    from unittest.mock import MagicMock
    class DummyFlashAttn:
        pass
    sys.modules["flash_attn"] = DummyFlashAttn
    sys.modules["flash_attn.layers"] = DummyFlashAttn
    sys.modules["flash_attn.layers.rotary"] = DummyFlashAttn
    DummyFlashAttn.RotaryEmbedding = MagicMock
except Exception:
    pass

# Deepeval mock patch
try:
    import sys
    import types
    from unittest.mock import MagicMock
    from importlib.machinery import ModuleSpec

    class MockFinderLoader:
        def __init__(self):
            self.modules = {}

        def find_spec(self, fullname, path, target=None):
            if fullname in ["deepeval", "trl", "outlines"] or fullname.startswith("deepeval.") or fullname.startswith("trl.") or fullname.startswith("outlines."):
                return ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            fullname = spec.name
            if fullname not in self.modules:
                class MockModule(types.ModuleType):
                    def __getattr__(self, name):
                        if name.startswith('__') and name.endswith('__'):
                            raise AttributeError(name)
                        return MagicMock
                mod = MockModule(fullname)
                mod.__path__ = []
                self.modules[fullname] = mod
                sys.modules[fullname] = mod
            return self.modules[fullname]

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, MockFinderLoader())
except Exception:
    pass
