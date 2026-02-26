
import sys

new_router_code = """
class SeqorthRouter(nn.Module):
    def __init__(self, config: SeqorthTextConfig, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.router_dim = config.router_dim
        self.top_k = config.num_experts_per_tok
        
        # [Simple & Robust]
        # Bias를 켜서 라우터가 스스로 전문가별 기본 확률을 학습하게 둠
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=True)
        
        # Initialization: Xavier Uniform (가장 안정적)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        # Dummy buffers for compatibility
        self.register_buffer("expert_load_ema", torch.zeros(self.num_experts), persistent=True)
        self.register_buffer("cv_ema", torch.tensor(1.0), persistent=True)
        self.ema_alpha = getattr(config, "ema_alpha", 0.99)
        
        # Dummy attributes for SeqorthMoE compatibility
        self.expression_projector = nn.Module()
        self.expression_projector.ortho_strength = 0.0

    def forward(self, x, hn, top_k=2, jitter_eps=0.01, step_frac=0.0, layer_idx: int = 0, **kwargs):
        # x: [Batch, Seq, Hidden]
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        
        # 1. [Input Normalization] LayerNorm
        x_norm = F.layer_norm(x_flat, (self.hidden_size,), eps=1e-5)
        
        # 2. [Gating] Logits
        logits = self.gate(x_norm) # [N, E]
        
        # 3. [Jitter] Noise Injection (Training Only)
        if self.training and jitter_eps > 0:
            noise = torch.rand_like(logits) * jitter_eps
            logits = logits + noise

        # 4. [Routing] Softmax & Top-K
        logits = logits.float()
        routing_weights = F.softmax(logits, dim=-1)
        
        k = top_k if top_k is not None else self.top_k
        topk_weight, topk_idx = torch.topk(routing_weights, k=k, dim=-1)
        
        # Renormalize (Sum to 1)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 5. [Loss Calculation] Local Aux Loss
        if self.training:
            p_mean = routing_weights.mean(0)
            # mask: [N, k, E]
            mask = F.one_hot(topk_idx, self.num_experts).float()
            # f_mean: [E] (average frequency)
            f_mean = mask.mean(dim=(0, 1))
            
            # Load Balancing Loss: N * sum(P * f) * coef
            lb_loss = (self.num_experts * (p_mean * f_mean).sum()) * 1.5
            
            # Orthogonality Loss
            w_norm = F.normalize(self.gate.weight, p=2, dim=1)
            gram = torch.matmul(w_norm, w_norm.t())
            identity = torch.eye(self.num_experts, device=x.device)
            ortho_loss = ((gram - identity) ** 2).mean() * 0.1
        else:
            lb_loss = torch.tensor(0.0, device=x.device)
            ortho_loss = torch.tensor(0.0, device=x.device)

        # 6. [Logging Data]
        routing_probs_full = routing_weights.view(batch_size, seq_len, self.num_experts)
        zero = torch.tensor(0.0, device=x.device)
        
        # 15 returns to match SeqorthMoE expectation
        return (
            topk_weight.view(batch_size, seq_len, k), # Multiplier
            topk_idx.view(batch_size, seq_len, k),    # Selected Experts
            None,               # Expression logits
            hn,                 # Context
            zero,               # Speciality
            zero,               # Cosine Sim
            zero,               # Contrastive
            routing_probs_full, # Logging
            zero,               # Expression Reg
            zero,               # Uncertainty
            zero,               # Entropy
            lb_loss,            # [Active] Load Balancing
            zero,               # Sinkhorn
            ortho_loss,         # [Active] Ortho
            zero                # Balance
        )
"""

file_path = "/home/conan/workspace/llm_training/models/seqorth_model.py"
import torch
import torch.nn as nn
import torch.nn.functional as F

# Just used for syntax check above, re-import not strictly needed for string manipulation

with open(file_path, "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "class SeqorthRouter(nn.Module):" in line:
        start_idx = i
    # Look for the start of SeqorthMoE to define end of Router
    # User file view shows:
    # 1970: iterations = 0
    # 1971: class SeqorthMoE(nn.Module):
    if "iterations = 0" in line and (i+1 < len(lines) and "class SeqorthMoE" in lines[i+1]):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    print(f"Found Router from line {start_idx} to {end_idx}")
    # Replace content
    new_lines = lines[:start_idx] + [new_router_code] + lines[end_idx:]
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("Successfully replaced SeqorthRouter")
else:
    print(f"Failed to find class boundaries. Start: {start_idx}, End: {end_idx}")
