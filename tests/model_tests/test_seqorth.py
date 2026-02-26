import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")  # Use a non-interactive backend for headless hosts.
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ==============================================================================
# 1. Newton-Schulz Projector (강제 직교화 모듈)
# ==============================================================================
class NewtonSchulzProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Orthogonal 초기화로 시작 (Newton-Schulz는 유지 보수 역할)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight)

    def newton_schulz(self, W, steps=20):
        """
        SVD 없는 반복적 직교화 (Differentiable)
        """
        # Frobenius Norm 기반 스케일 조정 (값이 너무 크면 발산 방지)
        norms = W.norm(p="fro")
        X = W / norms if norms > 1.0 else W

        # 행/열 중 짧은 쪽 기준으로 변환
        transpose = X.shape[0] < X.shape[1]
        if transpose:
            X = X.t()

        for _ in range(steps):
            A = torch.matmul(X.t(), X)
            X = 1.5 * X - 0.5 * torch.matmul(X, A)

        if transpose:
            X = X.t()
        return X

    def forward(self, x):
        # 1. 학습 중에는 매번 직교화된 가중치(W_ortho)를 생성
        # 2. Gradient는 W_ortho를 거쳐 원본 W로 흐름 (회전 방향 학습)
        W_ortho = self.newton_schulz(self.weight, steps=20)
        return F.linear(x, W_ortho)

    def get_ortho_error(self):
        with torch.no_grad():
            W = self.newton_schulz(self.weight, steps=20)
            if W.shape[0] <= W.shape[1]:
                gram = W @ W.T
            else:
                gram = W.T @ W
            identity = torch.eye(gram.shape[0], device=gram.device)
            return (gram - identity).norm().item()


# ==============================================================================
# 2. Sinkhorn Router (강제 배분 모듈)
# ==============================================================================
class SpecHornRouter(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # 직교화 Projector 사용
        self.projector = NewtonSchulzProjector(dim, num_experts * 16)  # RouterDim = 16

    def sinkhorn(self, cost, epsilon=0.05, iterations=3):
        # Log-space Sinkhorn
        Q = -cost / epsilon
        for _ in range(iterations):
            Q = Q - torch.logsumexp(Q, dim=-1, keepdim=True)  # Row Norm
            Q = Q - torch.logsumexp(Q, dim=0, keepdim=True)  # Col Norm (Simplified)
            # 실제 배분에서는 log(N/E) 보정이 필요하지만 여기선 단순화
        return torch.exp(Q)

    def forward(self, x):
        # 1. Projection (직교성 보장됨)
        # x: [B, D] -> proj: [B, E*R]
        proj = self.projector(x)
        B, _ = x.shape

        # 2. Reshape & Normalize
        # [B, E, R]
        router_vec = proj.view(B, self.num_experts, -1)
        router_vec = F.normalize(router_vec, p=2, dim=-1)

        # Input x도 비교를 위해 확장 (Toy example이라 x를 Query로 가정)
        input_vec = x.view(B, 1, -1).expand(-1, self.num_experts, -1)
        input_vec = input_vec[:, :, :16]  # 차원 맞추기용 slicing
        input_vec = F.normalize(input_vec, p=2, dim=-1)

        # 3. Distance Cost
        # cos_sim: [B, E]
        cos_sim = (router_vec * input_vec).sum(dim=-1)
        cost = 2 - 2 * cos_sim

        # 4. Sinkhorn Routing
        # Q: [B, E] (Sum rows=1, Sum cols~B/E)
        probs = self.sinkhorn(cost)

        return probs, cos_sim


# ==============================================================================
# 3. Main Verification Loop
# ==============================================================================
def run_test():
    BATCH_SIZE = 128
    DIM = 128
    NUM_EXPERTS = 8
    STEPS = 500
    ORTHO_PENALTY = 0.05  # 직교 유지 보강용
    BAL_PENALTY = 0.1     # 라우터 균형 유지 보강용

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}...")

    # 모델 & 옵티마이저
    router = SpecHornRouter(DIM, NUM_EXPERTS).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.001)  # LR 살짝 낮춤

    # 기록용
    history = {"loss": [], "cv": [], "ortho_error": []}

    print(
        f"{'Step':^6} | {'Task Loss':^10} | {'CV (Bal)':^10} | "
        f"{'Ortho Err':^10} | {'Max Prob':^10}"
    )
    print("-" * 60)

    for step in range(STEPS):
        # Dummy Input & Target
        # Task: 특정 패턴의 입력은 특정 전문가를 선호해야 함 (가정)
        x = torch.randn(BATCH_SIZE, DIM, device=device)

        # Forward
        probs, cos_sim = router(x)

        # Dummy Task Loss:
        # 그냥 max probability를 높이는 방향 (Entropy Minimization) -> 라우터가 확신을 갖도록 유도
        # 실제 MoE에서는 전문가의 Loss가 되겠지만, 여기선 라우팅 자체의 수렴성을 봅니다.
        # "가장 적합한(유사도가 높은) 전문가를 뽑아라"
        task_loss = -(probs * cos_sim.detach()).sum(dim=-1).mean()

        # 추가 페널티: 직교 유지 및 밸런스 유지
        W_ortho_train = router.projector.newton_schulz(router.projector.weight, steps=10)
        if W_ortho_train.shape[0] <= W_ortho_train.shape[1]:
            gram_train = W_ortho_train @ W_ortho_train.T
        else:
            gram_train = W_ortho_train.T @ W_ortho_train
        ortho_loss = (gram_train - torch.eye(gram_train.shape[0], device=device)).pow(2).mean()

        expert_usage = probs.mean(dim=0)
        balance_loss = (expert_usage - 1.0 / NUM_EXPERTS).pow(2).mean()

        total_loss = task_loss + ORTHO_PENALTY * ortho_loss + BAL_PENALTY * balance_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --- Metrics ---
        with torch.no_grad():
            # 1. CV (Load Balancing)
            expert_usage = probs.mean(dim=0)
            cv = expert_usage.std() / (expert_usage.mean() + 1e-6)

            # 2. Orthogonality
            ortho_err = router.projector.get_ortho_error()

            # 3. Max probability (확률 집중도)
            max_prob = probs.max().item()

            history["loss"].append(task_loss.item())
            history["cv"].append(cv.item())
            history["ortho_error"].append(ortho_err)

        if step % 50 == 0:
            print(
                f"{step:^6} | {task_loss.item():^10.4f} | {cv.item():^10.4f} | "
                f"{ortho_err:^10.6f} | {max_prob:^10.4f}"
            )

    return history


if __name__ == "__main__":
    torch.manual_seed(0)

    hist = run_test()

    if HAS_MATPLOTLIB:
        # 결과 시각화
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(hist["loss"], label="Task Loss")
        ax[0].set_title("Task Loss")
        ax[1].plot(hist["cv"], label="Expert CV", color="orange")
        ax[1].set_title("Expert CV (Target: ~0.0)")
        ax[2].plot(hist["ortho_error"], label="Ortho Error", color="green")
        ax[2].set_title("Orthogonality Error (Target: < 1e-4)")
        fig.tight_layout()

        out_path = os.path.join(os.getcwd(), "spechorn_results_fixed.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
    else:
        # Matplotlib이 없을 경우 지표만 저장
        out_path = os.path.join(os.getcwd(), "spechorn_history.pt")
        torch.save(hist, out_path)
        print(
            "matplotlib이 없어 그래프를 생략했습니다. "
            f"기록은 다음 경로에 저장했습니다: {out_path}"
        )

