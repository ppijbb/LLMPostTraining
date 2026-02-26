class SeqorthRouter(nn.Module):
    def __init__(self, config: SeqorthTextConfig, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.router_dim = config.router_dim

        # [SER Core 1] Spherical Linear Layer (No Bias)
        # 내적(Dot) 대신 코사인 유사도를 쓰기 위해 Bias를 제거하고 Weight만 사용
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # [SER Core 2] Exoskeleton PID Controller State (Buffers)
        # 학습 그라디언트와 무관하게 동작하는 외부 제어기 상태
        self.register_buffer("router_bias", torch.zeros(self.num_experts))       # b(t): 실제 주입될 편향
        self.register_buffer("accum_error", torch.zeros(self.num_experts))       # I-term: 누적 오차
        self.register_buffer("prev_error", torch.zeros(self.num_experts))        # D-term: 이전 오차
        
        # PID Hyperparameters (튜닝 가능)
        self.pid_kp = 0.05   # P: 즉각 반응
        self.pid_ki = 0.001  # I: 누적 불균형 해소
        self.pid_kd = 0.5    # D: 급격한 쏠림(MaxVio) 억제 (Damping)
        
        # Priority Head (Optional Intent)
        self.priority_head = nn.Linear(self.hidden_size + self.hidden_size, 1, bias=False)
        
        # Init
        nn.init.orthogonal_(self.gate.weight)
        nn.init.zeros_(self.priority_head.weight)

    def compute_orthogonality_loss(self):
        """
        [Seqorth Speciality] 전문가 벡터 간의 직교성을 강제하여 '유사한 전문가' 생성을 방지
        """
        # Weight Normalize
        w_norm = F.normalize(self.gate.weight, p=2, dim=1) # [E, D]
        
        # Gram Matrix (Cosine Similarity between Experts)
        # G_ij = cos(e_i, e_j)
        gram = torch.matmul(w_norm, w_norm.t()) # [E, E]
        
        # 대각선(자기 자신) 제외하고 0이 되도록 유도
        identity = torch.eye(self.num_experts, device=gram.device)
        ortho_loss = ((gram - identity) ** 2).mean()
        return ortho_loss

    def forward(self, x, hn, top_k=2, jitter_eps=0.01, step_frac=0.0, layer_idx: int = 0):
        """
        Spherical Exoskeleton Router (SER) Forward
        1. Spherical Projection: VLM Norm 문제 해결
        2. PID Bias Injection: CV/MaxVio 제어
        3. Orthogonality Loss: Speciality 확보
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        N = x_flat.size(0)
        E = self.num_experts
        
        # ========================================================================
        # [Phase 1] Spherical Projection (Geometry)
        # 이미지 토큰의 큰 Norm을 무시하고 '방향(의미)'만 봄
        # ========================================================================
        x_norm = F.normalize(x_flat, p=2, dim=-1)           # [N, D]
        w_norm = F.normalize(self.gate.weight, p=2, dim=-1) # [E, D]
        
        # Cosine Similarity Logits (-1.0 ~ 1.0)
        # Scale을 키워 Softmax의 변별력을 확보 (Temperature 역할)
        scale = 10.0 
        spherical_logits = F.linear(x_norm, w_norm) * scale # [N, E]

        # ========================================================================
        # [Phase 2] Exoskeleton PID Bias Injection
        # "인기 없는 놈은 점수를 더 주고(Bonus), 인기 있는 놈은 깎는다(Penalty)"
        # ========================================================================
        # Bias는 Gradient를 끊고(detach) 주입 -> 라우팅만 바꾸고, 전문가 학습은 방해 안 함
        
        # Priority (Intent)
        priority_score = 0.0
        if hn is not None:
            # hn 차원 맞추기 (Zero padding or projection)
            if hn.shape[-1] != self.hidden_size:
                 hn_proj = torch.zeros_like(x_flat) # Placeholder if dims don't match
            else:
                 hn_proj = hn.view(-1, self.hidden_size)
            
            divider_input = torch.cat([x_flat, hn_proj], dim=-1)
            priority_score = self.priority_head(divider_input) # [N, 1]

        # Final Logits = Semantic(Spherical) + System Bias(PID) + Context(Priority)
        routed_logits = spherical_logits + self.router_bias.unsqueeze(0) + priority_score

        # ========================================================================
        # [Phase 3] Standard Top-K (Soft Routing)
        # 무작위 채우기(Hard) 대신, 확률 기반 Top-K로 돌아가되 Bias로 제어
        # ========================================================================
        
        # 1. Routing Probabilities (Softmax)
        # 학습 시에는 노이즈 추가로 탐험 유도
        if self.training and jitter_eps > 0:
            routed_logits = routed_logits + torch.empty_like(routed_logits).uniform_(-jitter_eps, jitter_eps)
            
        routing_weights = F.softmax(routed_logits, dim=-1)
        
        # 2. Top-K Selection
        topk_weights, topk_indices = torch.topk(routing_weights, k=top_k, dim=-1)
        
        # 3. Normalize Weights (Sum to 1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # ========================================================================
        # [Phase 4] PID Controller Update (Feedback Loop)
        # "이번 배치의 부하를 보고, 다음 배치를 위한 편향(Bias)을 수정한다"
        # ========================================================================
        if self.training:
            with torch.no_grad():
                # 1. Load Measurement (Soft Load)
                # Hard Count 대신 Soft Probability 합을 사용하여 미세 조정
                current_load = routing_weights.sum(0) # [E]
                
                # [ZeRO-3] Global Sync (All-Reduce)
                if dist.is_initialized():
                    dist.all_reduce(current_load, op=dist.ReduceOp.SUM)
                    
                # 2. Error Calculation
                # 목표: 균등 분포 (Total Tokens * k / E)
                total_tokens = N * top_k
                if dist.is_initialized():
                    total_tokens *= dist.get_world_size()
                    
                target_load = total_tokens / E
                error = target_load - current_load # 양수면 과소(Boost 필요), 음수면 과다(Suppress 필요)
                
                # 3. PID Update
                # P: 즉각 반응
                delta = self.pid_kp * error
                
                # I: 누적 오차 (오래 굶은 놈 구제)
                self.accum_error.mul_(0.9).add_(error) # Decay factor 0.9 prevents integral windup
                delta += self.pid_ki * self.accum_error
                
                # D: 변화율 제어 (MaxVio 급증 방지)
                delta += self.pid_kd * (error - self.prev_error)
                self.prev_error.copy_(error)
                
                # Update Bias
                self.router_bias.add_(delta)
                
                # Safety Clamp (Bias가 너무 커져서 Semantic을 덮지 않도록)
                self.router_bias.clamp_(min=-10.0, max=10.0)

        # ========================================================================
        # [Phase 5] Outputs & Losses
        # ========================================================================
        
        # 1. Orthogonality Loss (Speciality)
        ortho_loss = self.compute_orthogonality_loss()
        
        # 2. Load Balancing Loss (Aux Loss Free)
        # PID가 제어하므로 별도 Loss 불필요 (0.0 반환)
        # 단, 모니터링을 위해 값은 계산해둘 수 있음
        
        # Return signature matching original
        # (weights, indices, ...)
        
        # For logging compatibility
        routing_probs_full = routing_weights 
        
        # Dummy zeros for unused losses
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        return (
            topk_weights,       # Multiplier
            topk_indices,       # Selected Experts
            None,               # Expression logits (unused)
            hn,                 # Context state
            zero,               # Speciality (Legacy)
            zero,               # Orthogonality (Domain - Legacy)
            zero,               # Contrastive
            routing_probs_full, # For logging
            zero,               # Expression Reg
            zero,               # Uncertainty
            zero,               # Entropy
            zero,               # Load Balancing (Legacy)
            zero,               # Sinkhorn
            ortho_loss,         # [New] Real Ortho Loss
            zero                # Balance Loss
        )
