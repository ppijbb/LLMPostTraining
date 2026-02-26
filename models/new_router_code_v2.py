class SeqorthRouter(nn.Module):
    def __init__(self, config: SeqorthTextConfig, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.router_dim = config.router_dim

        # [1] Pure Spherical Gate (No Bias in Linear, explicit learnable bias separate)
        # 가중치 초기화: Orthogonal하게 시작하여 초기 붕괴 방지
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        nn.init.orthogonal_(self.gate.weight)
        
        # [2] Minimalist Learnable Bias
        # 복잡한 제어기 대신, 모델이 알아서 학습할 수 있는 단순 파라미터
        self.bias = nn.Parameter(torch.zeros(self.num_experts))

        # [3] Intent Projection (Optional context awareness)
        self.priority_head = nn.Linear(self.hidden_size, 1, bias=False)
        nn.init.zeros_(self.priority_head.weight)

    def compute_ortho_loss(self):
        # 전문가 가중치 정규화
        w_norm = F.normalize(self.gate.weight, p=2, dim=1)
        # Gram Matrix: Experts 간의 코사인 유사도
        gram = torch.matmul(w_norm, w_norm.t())
        # 대각선(1)을 뺀 나머지(Off-diagonal)가 0이 되도록 강제
        identity = torch.eye(self.num_experts, device=gram.device)
        return ((gram - identity) ** 2).mean()

    def forward(self, x, hn, top_k=2, jitter_eps=0.01, step_frac=0.0, layer_idx: int = 0):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        # ----------------------------------------------------------------
        # 1. Spherical Logits Calculation (Computionally Precise)
        # ----------------------------------------------------------------
        # 입력과 전문가 가중치를 모두 정규화하여 '방향'만 비교
        x_norm = F.normalize(x_flat, p=2, dim=-1)           # [N, D]
        w_norm = F.normalize(self.gate.weight, p=2, dim=-1) # [E, D]
        
        # Cosine Similarity (Scale to restore gradient magnitude)
        # Temperature 10.0은 Softmax의 Sharpness를 보장함
        logits = F.linear(x_norm, w_norm) * 10.0 
        
        # Add Learnable Bias & Intent
        # hn(Context)이 있다면 Intent 점수 추가
        if hn is not None:
            # hn 차원 체크 및 투영 (필요시)
            hn_in = hn.view(-1, self.hidden_size) if hn.shape[-1] == self.hidden_size else torch.zeros_like(x_flat)
            intent_score = self.priority_head(hn_in)
            logits = logits + intent_score
            
        # Bias 더하기 (Broadcasting)
        logits = logits + self.bias

        # ----------------------------------------------------------------
        # 2. Standard Top-K Routing (Stable & Proven)
        # ----------------------------------------------------------------
        # Jitter for exploration (Training only)
        if self.training and jitter_eps > 0:
            logits = logits + torch.empty_like(logits).uniform_(-jitter_eps, jitter_eps)
            
        # Softmax & Top-K
        routing_weights = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(routing_weights, k=top_k, dim=-1)
        
        # Renormalize weights to sum to 1
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # ----------------------------------------------------------------
        # 3. Losses (Optimization Driver)
        # ----------------------------------------------------------------
        # Orthogonality Loss: 전문가가 겹치지 않게 강제 (Speciality)
        ortho_loss = self.compute_ortho_loss()
        
        # Load Balancing Loss: 한쪽으로 쏠리지 않게 유도 (Aux Loss)
        # (별도의 복잡한 함수 호출 대신 여기서 직접 계산하여 리턴)
        if self.training:
            # Importance: 이 배치에서 각 전문가에게 할당된 확률의 합
            importance = routing_weights.sum(0)
            # Load: 이 배치에서 각 전문가가 선택된 횟수 (Soft approximation)
            # Top-K 선택된 곳만 1, 나머지 0 인 마스크 근사
            load = torch.zeros_like(importance)
            load.scatter_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1), dtype=load.dtype))
            
            # Switch Transformer Style Loss: N * sum(P * f)
            # Normalize to stay strictly scalar
            num_tokens = x_flat.size(0)
            lb_loss = (importance * load).sum() * (self.num_experts / (num_tokens * top_k) ** 2)
        else:
            lb_loss = torch.tensor(0.0, device=x.device)

        # Unused outputs (Dummy placeholders for compatibility)
        zero = torch.tensor(0.0, device=x.device)
        
        return (
            topk_weight,        # Multiplier
            topk_idx,           # Selected Experts
            None,               # Expression logits
            hn,                 # Context state passed through
            zero,               # Speciality (Legacy)
            zero,               # Cosine Sim (Legacy)
            zero,               # Contrastive
            routing_weights,    # Full probs for logging
            zero,               # Expression Reg
            zero,               # Uncertainty
            zero,               # Entropy
            lb_loss,            # [Active] Load Balancing Loss
            zero,               # Sinkhorn
            ortho_loss,         # [Active] Orthogonality Loss
            zero                # Balance
        )
