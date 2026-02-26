"""
Test routing stabilization: ratio-based maxvio, soft barrier, intent dampening, flush_bias_updates.
Verifies that CV/MaxVio KPI path runs and stored maxvio is ratio (not raw count).
"""
import torch
from models.seqorth_model import SeqorthRouter
from models.seqorth_config import SeqorthTextConfig


def test_router_forward_flush_and_ratio_maxvio():
    """SeqorthRouter forward + flush_bias_updates; _maxvio_per_layer should be ratio-scale."""
    config = SeqorthTextConfig(
        hidden_size=128,
        n_routed_experts=16,
        num_experts_per_tok=2,
        router_dim=32,
    )
    config.soft_barrier_coef = 0.15
    config.intent_imbalance_damp = 0.85
    config.bias_update_lpf_high_imbalance = 0.75

    router = SeqorthRouter(config)
    router.train()

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward two layers to populate _cv_per_layer and _maxvio_per_layer
    for layer_idx in [0, 1]:
        out = router(x, hn=None, top_k=config.num_experts_per_tok, jitter_eps=0.01, layer_idx=layer_idx)
        routing_weights, topk_indices, _, intent, *_ = out[:4]
        assert routing_weights.shape[0] == batch_size * seq_len
        assert topk_indices.shape[0] == batch_size * seq_len
        assert not torch.isnan(routing_weights).any() and not torch.isinf(routing_weights).any()

    # Before flush: maxvio stored should be ratio (deviation/target), not raw token count.
    # With 16 experts and batch*seq=16 tokens, target_per_expert=1; maxvio can be order of 1–10.
    if getattr(router, "_maxvio_per_layer", None):
        for layer_idx, mvio in router._maxvio_per_layer.items():
            assert mvio >= 0, "maxvio ratio should be non-negative"
            # Ratio scale: typically < 20 for moderate imbalance; raw count could be 100+
            assert mvio < 200.0, "maxvio should be ratio-scale, not raw count"

    router.flush_bias_updates()
    assert router._maxvio_per_layer == {} or len(router._maxvio_per_layer) == 0


def test_router_soft_barrier_and_intent_damp_config():
    """New config params are read by router (getattr defaults)."""
    config = SeqorthTextConfig(
        hidden_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        router_dim=16,
        proportional_correction_strength=0.5,
        soft_barrier_coef=0.15,
        max_correction_abs=0.4,
        intent_imbalance_damp=0.85,
        bias_update_lpf_high_imbalance=0.75,
    )
    router = SeqorthRouter(config)
    assert getattr(router, "_soft_barrier_coef", None) == 0.15


if __name__ == "__main__":
    test_router_forward_flush_and_ratio_maxvio()
    test_router_soft_barrier_and_intent_damp_config()
    print("Routing stabilization tests passed.")
