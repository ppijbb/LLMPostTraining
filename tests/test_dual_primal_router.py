"""
Dual-Primal Global Router Test
Implements Online Sinkhorn via Dual Variable Optimization (Lagrange Multiplier Method)

Theory:
  Maximize H(P) + <S, P>  s.t.  sum(P, dim=0) = T/E
  Lagrangian: L(P, λ) = H(P) + <S + λ, P> - λ(T/E)

Mechanism:
  1. Primal Step (Forward): P = softmax(Logits + λ)
  2. Dual Step (Update): λ += η * (Target_Load - Current_Load)

Result:
  CV converges to 0.00x theoretically and practically because λ acts as an
  infinite-gain integral controller enforcing mass conservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.seqorth_config import SeqorthTextConfig


class DualPrimalRouter(nn.Module):
    """
    [Dual-Primal Global Router]
    Implements Online Sinkhorn via Dual Variable Optimization.
    """
    def __init__(self, config: SeqorthTextConfig, use_momentum=False, **kwargs):
        super().__init__()
        self.config = config
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.router_dim = getattr(config, "router_dim", 64)
        self.use_momentum = use_momentum
        
        # 1. Basis (Semantic)
        self.B = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        nn.init.orthogonal_(self.B)
        
        # 2. [CORE] Expert Dual Variable (λ)
        # This is NOT a learnable parameter (no gradient descent).
        # It is updated via the Dual Ascent step (Mass Correction Rule).
        # Must be FP32 to prevent underflow accumulation.
        self.register_buffer('dual_lambda', torch.zeros(self.num_experts, dtype=torch.float32), persistent=True)
        
        # 3. [NEW] Momentum Buffer (for noise filtering)
        if self.use_momentum:
            self.register_buffer('dual_error_ema', torch.zeros(self.num_experts, dtype=torch.float32), persistent=True)
            self.momentum_beta = getattr(config, "momentum_beta", 0.1)  # Smoothing factor (0.1 = 90% history retention)
        else:
            self.momentum_beta = None
        
        # 4. Hyperparameters
        self.dual_learning_rate = getattr(config, "dual_learning_rate", 0.1)  # η (Correction Strength)
        self.target_load = 1.0 / self.num_experts
        
        # 5. Utils
        self.top_k = min(getattr(config, "num_experts_per_tok", 2), self.num_experts)
        self.input_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        
        # Tracking
        self.register_buffer("expert_load_ema", torch.zeros(self.num_experts), persistent=True)
        self.register_buffer("cv_ema", torch.tensor(1.0), persistent=True)
        self.ema_alpha = 0.99
        
        # Backward hook for Dual Update
        if self.use_momentum:
            self.register_full_backward_hook(self._momentum_dual_update_hook)
        else:
            self.register_full_backward_hook(self._dual_update_hook)

    def _dual_update_hook(self, module, grad_input, grad_output):
        """
        [Vanilla Dual Update] Direct Integral Action
        λ ← λ + η * (1/E - m_i)
        Susceptible to batch noise in small batch sizes.
        """
        if not self.training:
            return
        
        with torch.no_grad():
            # Retrieve cached probabilities from forward pass
            if hasattr(self, 'last_probs'):
                P = self.last_probs  # [Batch, Seq, Experts]
                
                # Calculate Global Mass Error (Current Load)
                # m_i = (1/T) * sum_t p_{t,i}
                current_mass = P.mean(dim=(0, 1))  # [Experts]
                
                # Update EMA for monitoring
                self.expert_load_ema.mul_(self.ema_alpha).add_(current_mass, alpha=1.0 - self.ema_alpha)
                
                # Mass Conservation Constraint Error
                # Error > 0: Underloaded (Need more mass) -> Increase λ
                # Error < 0: Overloaded (Too much mass) -> Decrease λ
                instant_error = self.target_load - current_mass
                
                # Dual Ascent Step (Direct Integral Action - No Filtering)
                # Accumulates strictly until error is eliminated.
                self.dual_lambda += self.dual_learning_rate * instant_error
                
                # Center λ to prevent drift (optional, keeps logic stable)
                self.dual_lambda -= self.dual_lambda.mean()
                
                # Update CV for monitoring
                load = self.expert_load_ema
                var_p = load.var(unbiased=False)
                mean_p = load.mean()
                current_cv = (var_p.sqrt() / (mean_p + 1e-6))
                self.cv_ema.mul_(self.ema_alpha).add_(current_cv, alpha=1.0 - self.ema_alpha)
                
                # Clear cache
                self.last_probs = None

    def _momentum_dual_update_hook(self, module, grad_input, grad_output):
        """
        [Momentum Dual Update] Filtered Integral Action
        1. Measure instant error (Noisy)
        2. Smooth error via EMA (Filter)
        3. Update Lambda using smoothed error (Stable)
        
        This suppresses batch noise and allows convergence even with small batch sizes.
        """
        if not self.training:
            return
        
        with torch.no_grad():
            # Retrieve cached probabilities from forward pass
            if hasattr(self, 'last_probs'):
                P = self.last_probs  # [Batch, Seq, Experts]
                
                # Calculate Global Mass Error (Current Load)
                # m_i = (1/T) * sum_t p_{t,i}
                current_mass = P.mean(dim=(0, 1)).float()  # [Experts]
                
                # Update EMA for monitoring
                self.expert_load_ema.mul_(self.ema_alpha).add_(current_mass, alpha=1.0 - self.ema_alpha)
                
                # 1. Instant Mass Error (Contains high variance noise from small batch)
                instant_error = self.target_load - current_mass
                
                # 2. Momentum Filter (Low-pass filter)
                # ema_t = (1-beta)*ema_{t-1} + beta*error_t
                # This filters out high-frequency batch noise
                self.dual_error_ema.mul_(1.0 - self.momentum_beta).add_(instant_error * self.momentum_beta)
                
                # 3. Stable Dual Ascent (Integral Action on Filtered Signal)
                # Update lambda using the smoothed error signal
                # This is more stable than vanilla because it responds to trends, not noise
                self.dual_lambda += self.dual_learning_rate * self.dual_error_ema
                
                # Center λ to prevent drift (optional, keeps logic stable)
                self.dual_lambda -= self.dual_lambda.mean()
                
                # Update CV for monitoring
                load = self.expert_load_ema
                var_p = load.var(unbiased=False)
                mean_p = load.mean()
                current_cv = (var_p.sqrt() / (mean_p + 1e-6))
                self.cv_ema.mul_(self.ema_alpha).add_(current_cv, alpha=1.0 - self.ema_alpha)
                
                # Clear cache
                self.last_probs = None

    def forward(self, x, hn=None, top_k=2, jitter_eps=0.01, step_frac=0.0, layer_idx=0, **kwargs):
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        x_norm = self.input_norm(x_flat)
        
        # [Step 1] Semantic Logits: s_{t,i}
        B_norm = F.normalize(self.B, p=2, dim=1)
        logits = F.linear(F.normalize(x_norm, p=2, dim=1), B_norm)  # Cosine Sim
        
        # Add jitter noise during training
        if self.training and jitter_eps > 0:
            noise = torch.empty_like(logits).uniform_(-jitter_eps, jitter_eps)
            logits = logits + noise
        
        # [Step 2] Global Dual Correction: s'_{t,i} = s_{t,i} + λ_i
        # λ acts as the column normalization factor from Sinkhorn
        # detaching λ ensures it's treated as a constant during Backprop (Primal step)
        corrected_logits = logits + self.dual_lambda.unsqueeze(0).to(logits.dtype)
        
        # [Step 3] Softmax (Primal Solution)
        # p_{t,i} = exp(s'_{t,i}) / Z
        probs = F.softmax(corrected_logits, dim=-1)
        
        # Dispatch (Top-k)
        top_k_probs, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize weights for computation
        multiplier = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        multiplier = multiplier.view(batch, seq, self.top_k)
        selected_experts = selected_experts.view(batch, seq, self.top_k)
        
        # Cache for Dual Update
        if self.training:
            self.last_probs = probs.detach()
            
        # Optional: Ortho Loss for Semantic Diversity (Does not affect mass balance)
        if self.training:
            gram = B_norm @ B_norm.t()
            eye = torch.eye(self.num_experts, device=x.device, dtype=gram.dtype)
            ortho_loss = ((gram - eye) ** 2).mean()
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)

        # Output formatting (matching SeqorthRouter interface)
        zero = torch.tensor(0.0, device=x.device, requires_grad=True)
        return (
            multiplier, selected_experts, None, probs,
            zero, zero, ortho_loss, zero, zero, zero
        )


def verify_dual_primal_router(use_momentum=False, batch_size=128):
    """
    Comprehensive verification test for Dual-Primal Router
    
    Args:
        use_momentum: If True, use momentum filtering (recommended for small batch sizes)
        batch_size: Batch size (smaller = more noise, momentum helps more)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Config
    BATCH = batch_size
    SEQ = 1
    DIM = 128
    EXPERTS = 64
    STEPS = 2000
    
    router_type = "Momentum Dual-Primal" if use_momentum else "Vanilla Dual-Primal"
    print("=" * 80)
    print(f"{router_type} Router Verification Test")
    print("=" * 80)
    print(f"Config: BATCH={BATCH}, SEQ={SEQ}, DIM={DIM}, EXPERTS={EXPERTS}, STEPS={STEPS}")
    print(f"Momentum: {use_momentum}")
    print()
    
    # Create config
    config = SeqorthTextConfig()
    config.n_routed_experts = EXPERTS
    config.hidden_size = DIM
    config.router_dim = 64
    config.num_experts_per_tok = 2
    config.dual_learning_rate = 0.1  # η (Correction Strength)
    if use_momentum:
        config.momentum_beta = 0.1  # Smoothing factor (0.1 = 90% history retention)
    
    # Initialize router
    model = DualPrimalRouter(config, use_momentum=use_momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    # Tracking
    cv_history = []
    ortho_loss_history = []
    expert_usage_history = []
    lambda_history = []
    inf_nan_detected = False
    
    print("Starting verification...")
    print()
    
    # Simulate MODERATE skew (challenging but realistic)
    # For small batch sizes, this creates significant noise
    center = torch.randn(1, DIM)
    skew_factor = 0.3  # Moderate skew: 30% center, 70% noise
    
    for i in range(STEPS):
        # Generate data
        noise = torch.randn(BATCH, SEQ, DIM)
        x = center * skew_factor + noise * (1 - skew_factor)
        
        try:
            # Forward pass
            router_output = model(
                x,
                hn=None,
                top_k=config.num_experts_per_tok,
                jitter_eps=0.01,
                step_frac=i / STEPS,
                layer_idx=0
            )
            
            # Unpack output
            multiplier, selected_experts, hn_next, routing_probs_full, \
                speciality_loss, cosine_similarities, ortho_loss, \
                entropy_loss, routing_uncertainty, contrastive_loss = router_output
            
            # Check for Inf/NaN
            if torch.isinf(ortho_loss) or torch.isnan(ortho_loss):
                print(f"❌ FAILED: Ortho loss is {ortho_loss.item()} at step {i}")
                inf_nan_detected = True
                break
            
            # Get CV from router state
            cv = model.cv_ema.item()
            
            # Track lambda values
            lambda_vals = model.dual_lambda.detach().cpu().numpy()
            lambda_history.append(lambda_vals.copy())
            
            # Track expert usage
            if routing_probs_full is not None:
                expert_usage = routing_probs_full.mean(dim=(0, 1)).detach().cpu().numpy()
                expert_usage_history.append(expert_usage)
            
            # Record history
            cv_history.append(cv)
            ortho_loss_history.append(ortho_loss.item())
            
            # Optimize
            # Include routing_probs_full and selected_experts in loss to trigger backward hook
            routing_loss = routing_probs_full.mean() * 0.001
            selected_loss = selected_experts.float().mean() * 0.001
            loss = ortho_loss * 0.1 + routing_loss + selected_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Progress update
            if (i + 1) % 50 == 0:
                lambda_std = lambda_vals.std()
                lambda_mean = lambda_vals.mean()
                print(f"Step {i+1}/{STEPS}: CV={cv:.6f}, OrthoLoss={ortho_loss.item():.6f}, "
                      f"Lambda(mean={lambda_mean:.4f}, std={lambda_std:.4f})")
        
        except Exception as e:
            print(f"❌ FAILED: Exception at step {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    print("=" * 80)
    print("Verification Results")
    print("=" * 80)
    
    # Check results
    final_cv = cv_history[-1] if cv_history else float('inf')
    max_ortho_loss = max(ortho_loss_history) if ortho_loss_history else float('inf')
    final_ortho_loss = ortho_loss_history[-1] if ortho_loss_history else float('inf')
    
    # Check expert usage
    if expert_usage_history:
        final_expert_usage = expert_usage_history[-1]
        min_expert_usage = final_expert_usage.min()
        max_expert_usage = final_expert_usage.max()
        unused_experts = (final_expert_usage < 1e-6).sum()
    else:
        min_expert_usage = 0.0
        max_expert_usage = 0.0
        unused_experts = EXPERTS
    
    # Print results
    print(f"Final CV: {final_cv:.6f} (Target: < 0.005, Preferably < 0.0001)")
    print(f"Max Ortho Loss: {max_ortho_loss:.6f}")
    print(f"Final Ortho Loss: {final_ortho_loss:.6f}")
    print(f"Min Expert Usage: {min_expert_usage:.6f}")
    print(f"Max Expert Usage: {max_expert_usage:.6f}")
    print(f"Unused Experts: {unused_experts}/{EXPERTS}")
    print(f"Inf/NaN Detected: {inf_nan_detected}")
    print()
    
    # Final lambda stats
    if lambda_history:
        final_lambda = lambda_history[-1]
        print(f"Final Lambda Stats:")
        print(f"  Mean: {final_lambda.mean():.6f}")
        print(f"  Std: {final_lambda.std():.6f}")
        print(f"  Min: {final_lambda.min():.6f}")
        print(f"  Max: {final_lambda.max():.6f}")
        print()
    
    # Verification checks
    checks_passed = 0
    total_checks = 4
    
    # Check 1: CV < 0.005
    if final_cv < 0.005:
        print("✅ PASS: CV < 0.005")
        checks_passed += 1
    else:
        print(f"❌ FAIL: CV >= 0.005 (got {final_cv:.6f})")
    
    # Check 2: No Inf/NaN
    if not inf_nan_detected:
        print("✅ PASS: No Inf/NaN detected")
        checks_passed += 1
    else:
        print("❌ FAIL: Inf/NaN detected")
    
    # Check 3: All experts used
    if unused_experts == 0:
        print("✅ PASS: All experts are used")
        checks_passed += 1
    else:
        print(f"❌ FAIL: {unused_experts} experts unused")
    
    # Check 4: CV convergence trend
    if len(cv_history) >= 100:
        recent_cv = np.mean(cv_history[-100:])
        early_cv = np.mean(cv_history[:100])
        if recent_cv < early_cv * 0.5:  # CV decreased by at least 50%
            print(f"✅ PASS: CV convergence trend (early: {early_cv:.6f}, recent: {recent_cv:.6f})")
            checks_passed += 1
        else:
            print(f"⚠️  WARN: CV convergence trend weak (early: {early_cv:.6f}, recent: {recent_cv:.6f})")
    else:
        print("⚠️  WARN: Not enough steps to check convergence trend")
    
    print()
    print(f"Checks Passed: {checks_passed}/{total_checks}")
    print()
    
    # Plot results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # CV history
        axes[0, 0].plot(cv_history)
        axes[0, 0].axhline(y=0.005, color='r', linestyle='--', label='Target (0.005)')
        axes[0, 0].axhline(y=0.0001, color='g', linestyle='--', label='Preferable (0.0001)')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('CV (Coefficient of Variation)')
        axes[0, 0].set_title('CV Convergence (Target < 0.005)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Ortho loss history
        axes[0, 1].plot(ortho_loss_history)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Ortho Loss')
        axes[0, 1].set_title('Orthogonality Loss')
        axes[0, 1].grid(True)
        
        # Lambda evolution (last 100 steps)
        if lambda_history:
            lambda_array = np.array(lambda_history[-100:])
            axes[1, 0].plot(lambda_array)
            axes[1, 0].set_xlabel('Step (last 100)')
            axes[1, 0].set_ylabel('Lambda Value')
            axes[1, 0].set_title('Dual Variable (λ) Evolution')
            axes[1, 0].grid(True)
        
        # Expert usage distribution (final step)
        if expert_usage_history:
            axes[1, 1].bar(range(EXPERTS), final_expert_usage)
            axes[1, 1].axhline(y=1.0/EXPERTS, color='r', linestyle='--', label='Target (1/E)')
            axes[1, 1].set_xlabel('Expert Index')
            axes[1, 1].set_ylabel('Usage Probability')
            axes[1, 1].set_title(f'Expert Usage Distribution (Step {STEPS})')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        
        plt.tight_layout()
        suffix = "momentum" if use_momentum else "vanilla"
        output_path = project_root / f'dual_primal_router_verification_{suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plot: {e}")
    
    # Final verdict
    success = (checks_passed >= 3) and (final_cv < 0.005) and (not inf_nan_detected)
    
    if success:
        print("=" * 80)
        print("✅ VERIFICATION PASSED")
        print("=" * 80)
        return True
    else:
        print("=" * 80)
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        return False


def compare_vanilla_vs_momentum():
    """
    Direct comparison: Vanilla vs Momentum with small batch size (128)
    This demonstrates the noise filtering effect of momentum.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    BATCH = 128  # Fixed small batch (high noise environment)
    SEQ = 1
    DIM = 128
    EXPERTS = 64
    STEPS = 1000
    
    print("=" * 80)
    print("Vanilla vs Momentum Comparison (Batch Size 128 Fixed)")
    print("=" * 80)
    print(f"Config: BATCH={BATCH}, EXPERTS={EXPERTS}, STEPS={STEPS}")
    print()
    
    # Create configs
    config = SeqorthTextConfig()
    config.n_routed_experts = EXPERTS
    config.hidden_size = DIM
    config.router_dim = 64
    config.num_experts_per_tok = 2
    config.dual_learning_rate = 0.1
    config.momentum_beta = 0.1
    
    # Initialize both routers
    vanilla = DualPrimalRouter(config, use_momentum=False)
    momentum = DualPrimalRouter(config, use_momentum=True)
    
    # Sync initialization for fair comparison
    momentum.B.data.copy_(vanilla.B.data)
    
    vanilla_optimizer = torch.optim.Adam(vanilla.parameters(), lr=0.001)
    momentum_optimizer = torch.optim.Adam(momentum.parameters(), lr=0.001)
    
    vanilla.train()
    momentum.train()
    
    # Tracking
    results = {
        'Vanilla': {'cv': [], 'lambda_std': []},
        'Momentum': {'cv': [], 'lambda_std': []}
    }
    
    # Data generator (moderate skew)
    center = torch.randn(1, DIM)
    skew_factor = 0.3
    
    print("Running comparison...")
    print()
    
    for step in range(STEPS):
        # Generate data
        noise = torch.randn(BATCH, SEQ, DIM)
        x = center * skew_factor + noise * (1 - skew_factor)
        
        # Vanilla run
        v_output = vanilla(x, hn=None, top_k=2, jitter_eps=0.01, step_frac=step/STEPS, layer_idx=0)
        v_multiplier, v_selected, _, v_probs, _, _, v_ortho, _, _, _ = v_output
        v_loss = v_ortho * 0.1 + v_probs.mean() * 0.001 + v_selected.float().mean() * 0.001
        vanilla_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(vanilla.parameters(), max_norm=1.0)
        vanilla_optimizer.step()
        
        # Momentum run
        m_output = momentum(x, hn=None, top_k=2, jitter_eps=0.01, step_frac=step/STEPS, layer_idx=0)
        m_multiplier, m_selected, _, m_probs, _, _, m_ortho, _, _, _ = m_output
        m_loss = m_ortho * 0.1 + m_probs.mean() * 0.001 + m_selected.float().mean() * 0.001
        momentum_optimizer.zero_grad()
        m_loss.backward()
        torch.nn.utils.clip_grad_norm_(momentum.parameters(), max_norm=1.0)
        momentum_optimizer.step()
        
        # Record metrics
        with torch.no_grad():
            v_cv = vanilla.cv_ema.item()
            m_cv = momentum.cv_ema.item()
            v_lambda_std = vanilla.dual_lambda.std().item()
            m_lambda_std = momentum.dual_lambda.std().item()
            
            results['Vanilla']['cv'].append(v_cv)
            results['Vanilla']['lambda_std'].append(v_lambda_std)
            results['Momentum']['cv'].append(m_cv)
            results['Momentum']['lambda_std'].append(m_lambda_std)
            
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{STEPS}:")
                print(f"  Vanilla  - CV: {v_cv:.6f}, Lambda Std: {v_lambda_std:.6f}")
                print(f"  Momentum - CV: {m_cv:.6f}, Lambda Std: {m_lambda_std:.6f}")
                print()
    
    # Final results
    print("=" * 80)
    print("Comparison Results")
    print("=" * 80)
    v_final_cv = results['Vanilla']['cv'][-1]
    m_final_cv = results['Momentum']['cv'][-1]
    
    print(f"Final CV (Vanilla):  {v_final_cv:.6f}")
    print(f"Final CV (Momentum): {m_final_cv:.6f}")
    print(f"Improvement: {v_final_cv / m_final_cv:.2f}x better" if m_final_cv > 0 else "Momentum achieved perfect balance")
    print()
    
    # Plot comparison
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # CV comparison
        axes[0].plot(results['Vanilla']['cv'], label='Vanilla Dual (Noisy)', alpha=0.7, color='gray', linewidth=1.5)
        axes[0].plot(results['Momentum']['cv'], label='Momentum Dual (Smoothed)', linewidth=2, color='red')
        axes[0].axhline(0.005, color='green', linestyle='--', label='Target 0.005')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('CV (Log Scale)')
        axes[0].set_title('CV Convergence: Vanilla vs Momentum (Batch Size 128)')
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)
        
        # Lambda std comparison (shows stability)
        axes[1].plot(results['Vanilla']['lambda_std'], label='Vanilla Lambda Std', alpha=0.7, color='gray', linewidth=1.5)
        axes[1].plot(results['Momentum']['lambda_std'], label='Momentum Lambda Std', linewidth=2, color='red')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Lambda Standard Deviation')
        axes[1].set_title('Dual Variable Stability (Lower = More Stable)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = project_root / 'momentum_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not generate comparison plot: {e}")
    
    # Success criteria
    momentum_success = m_final_cv < 0.005
    vanilla_success = v_final_cv < 0.005
    
    print("=" * 80)
    if momentum_success:
        print("✅ Momentum PASSED: CV < 0.005")
    else:
        print(f"❌ Momentum FAILED: CV = {m_final_cv:.6f} >= 0.005")
    
    if vanilla_success:
        print("✅ Vanilla PASSED: CV < 0.005")
    else:
        print(f"❌ Vanilla FAILED: CV = {v_final_cv:.6f} >= 0.005")
    print("=" * 80)
    
    return momentum_success


def test_forward_ema_duplication_bug():
    """
    [CRITICAL BUG REPRODUCTION TEST]
    
    This test reproduces the bug in seqorth_model.py where CV EMA is updated
    BOTH in forward pass AND backward hook, causing CV to explode.
    
    Problem:
    - seqorth_model.py updates cv_ema and expert_load_ema in forward pass (line 1961, 1964)
    - Then updates them again in backward hook (line 1451, 1474)
    - With gradient checkpointing, forward pass runs multiple times, causing
      multiple EMA updates per backward pass, leading to incorrect CV calculation
    
    This test demonstrates the difference between:
    1. Correct: Only update EMA in backward hook (like test_dual_primal_router.py)
    2. Buggy: Update EMA in both forward and backward (like seqorth_model.py)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    BATCH = 128
    SEQ = 1
    DIM = 128
    EXPERTS = 64
    STEPS = 1000
    
    print("=" * 80)
    print("Forward EMA Duplication Bug Reproduction Test")
    print("=" * 80)
    print("This test demonstrates why seqorth_model.py CV explodes:")
    print("  - Forward pass updates EMA (runs multiple times with checkpointing)")
    print("  - Backward hook also updates EMA")
    print("  - Result: EMA updated multiple times per step -> CV calculation wrong")
    print()
    
    # Create config
    config = SeqorthTextConfig()
    config.n_routed_experts = EXPERTS
    config.hidden_size = DIM
    config.router_dim = 64
    config.num_experts_per_tok = 2
    config.dual_learning_rate = 0.1
    config.momentum_beta = 0.1
    
    # Create two routers: one correct, one buggy
    class BuggyDualPrimalRouter(DualPrimalRouter):
        """
        Buggy version that updates EMA in forward pass (like seqorth_model.py)
        """
        def forward(self, x, hn=None, top_k=2, jitter_eps=0.01, step_frac=0.0, layer_idx=0, **kwargs):
            batch, seq, dim = x.shape
            x_flat = x.view(-1, dim)
            x_norm = self.input_norm(x_flat)
            
            # [Step 1] Semantic Logits
            B_norm = F.normalize(self.B, p=2, dim=1)
            logits = F.linear(F.normalize(x_norm, p=2, dim=1), B_norm)
            
            # Add jitter noise during training
            if self.training and jitter_eps > 0:
                noise = torch.empty_like(logits).uniform_(-jitter_eps, jitter_eps)
                logits = logits + noise
            
            # [Step 2] Global Dual Correction
            corrected_logits = logits + self.dual_lambda.unsqueeze(0).to(logits.dtype)
            
            # [Step 3] Softmax
            probs = F.softmax(corrected_logits, dim=-1)
            routing_probs_full = probs.view(batch, seq, self.num_experts)
            
            # [BUG] Update EMA in forward pass (like seqorth_model.py line 1961, 1964)
            if self.training:
                with torch.no_grad():
                    current_load = routing_probs_full.detach().mean(dim=(0, 1))
                    
                    # Store for backward hook
                    self.last_probs = routing_probs_full.detach()
                    
                    # [BUG] Update EMA in forward (this runs multiple times with checkpointing!)
                    var_p = current_load.var(unbiased=False)
                    mean_p = current_load.mean()
                    current_cv = (var_p.sqrt() / (mean_p + 1e-6))
                    self.cv_ema.mul_(self.ema_alpha).add_(current_cv * (1.0 - self.ema_alpha))
                    self.expert_load_ema.mul_(self.ema_alpha).add_(current_load, alpha=1.0 - self.ema_alpha)
            
            # Dispatch (Top-k)
            top_k_probs, selected_experts = torch.topk(probs, self.top_k, dim=-1)
            multiplier = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
            multiplier = multiplier.view(batch, seq, self.top_k)
            selected_experts = selected_experts.view(batch, seq, self.top_k)
            
            # Optional: Ortho Loss
            if self.training:
                gram = B_norm @ B_norm.t()
                eye = torch.eye(self.num_experts, device=x.device, dtype=gram.dtype)
                ortho_loss = ((gram - eye) ** 2).mean()
            else:
                ortho_loss = torch.tensor(0.0, device=x.device)
            
            zero = torch.tensor(0.0, device=x.device, requires_grad=True)
            return (
                multiplier, selected_experts, None, routing_probs_full,
                zero, zero, ortho_loss, zero, zero, zero
            )
    
    # Initialize routers
    correct_router = DualPrimalRouter(config, use_momentum=True)
    buggy_router = BuggyDualPrimalRouter(config, use_momentum=True)
    
    # Sync initialization
    buggy_router.B.data.copy_(correct_router.B.data)
    buggy_router.dual_lambda.copy_(correct_router.dual_lambda)
    buggy_router.dual_error_ema.copy_(correct_router.dual_error_ema)
    buggy_router.expert_load_ema.copy_(correct_router.expert_load_ema)
    buggy_router.cv_ema.copy_(correct_router.cv_ema)
    
    correct_optimizer = torch.optim.Adam(correct_router.parameters(), lr=0.001)
    buggy_optimizer = torch.optim.Adam(buggy_router.parameters(), lr=0.001)
    
    correct_router.train()
    buggy_router.train()
    
    # Tracking
    results = {
        'Correct': {'cv': [], 'lambda_std': [], 'expert_load_std': []},
        'Buggy': {'cv': [], 'lambda_std': [], 'expert_load_std': []}
    }
    
    # Data generator
    center = torch.randn(1, DIM)
    skew_factor = 0.3
    
    print("Running comparison (Correct vs Buggy)...")
    print()
    
    for step in range(STEPS):
        # Generate data
        noise = torch.randn(BATCH, SEQ, DIM)
        x = center * skew_factor + noise * (1 - skew_factor)
        
        # Simulate gradient checkpointing: forward pass runs multiple times
        # In real training with checkpointing, forward runs 2-3 times per backward
        num_forward_passes = 2  # Simulate checkpointing
        
        # Correct router (only updates EMA in backward)
        for _ in range(num_forward_passes):
            c_output = correct_router(x, hn=None, top_k=2, jitter_eps=0.01, step_frac=step/STEPS, layer_idx=0)
        c_multiplier, c_selected, _, c_probs, _, _, c_ortho, _, _, _ = c_output
        c_loss = c_ortho * 0.1 + c_probs.mean() * 0.001 + c_selected.float().mean() * 0.001
        correct_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(correct_router.parameters(), max_norm=1.0)
        correct_optimizer.step()
        
        # Buggy router (updates EMA in forward AND backward)
        for _ in range(num_forward_passes):
            b_output = buggy_router(x, hn=None, top_k=2, jitter_eps=0.01, step_frac=step/STEPS, layer_idx=0)
        b_multiplier, b_selected, _, b_probs, _, _, b_ortho, _, _, _ = b_output
        b_loss = b_ortho * 0.1 + b_probs.mean() * 0.001 + b_selected.float().mean() * 0.001
        buggy_optimizer.zero_grad()
        b_loss.backward()
        torch.nn.utils.clip_grad_norm_(buggy_router.parameters(), max_norm=1.0)
        buggy_optimizer.step()
        
        # Record metrics
        with torch.no_grad():
            c_cv = correct_router.cv_ema.item()
            b_cv = buggy_router.cv_ema.item()
            c_lambda_std = correct_router.dual_lambda.std().item()
            b_lambda_std = buggy_router.dual_lambda.std().item()
            c_load_std = correct_router.expert_load_ema.std().item()
            b_load_std = buggy_router.expert_load_ema.std().item()
            
            results['Correct']['cv'].append(c_cv)
            results['Correct']['lambda_std'].append(c_lambda_std)
            results['Correct']['expert_load_std'].append(c_load_std)
            results['Buggy']['cv'].append(b_cv)
            results['Buggy']['lambda_std'].append(b_lambda_std)
            results['Buggy']['expert_load_std'].append(b_load_std)
            
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{STEPS}:")
                print(f"  Correct - CV: {c_cv:.6f}, Lambda Std: {c_lambda_std:.6f}, Load Std: {c_load_std:.6f}")
                print(f"  Buggy   - CV: {b_cv:.6f}, Lambda Std: {b_lambda_std:.6f}, Load Std: {b_load_std:.6f}")
                print(f"  CV Ratio (Buggy/Correct): {b_cv / c_cv:.2f}x")
                print()
    
    # Final results
    print("=" * 80)
    print("Bug Reproduction Results")
    print("=" * 80)
    c_final_cv = results['Correct']['cv'][-1]
    b_final_cv = results['Buggy']['cv'][-1]
    
    print(f"Final CV (Correct): {c_final_cv:.6f}")
    print(f"Final CV (Buggy):   {b_final_cv:.6f}")
    print(f"CV Explosion Factor: {b_final_cv / c_final_cv:.2f}x")
    print()
    
    if b_final_cv > c_final_cv * 1.5:
        print("✅ BUG CONFIRMED: Buggy version has significantly higher CV")
        print("   This explains why seqorth_model.py CV explodes during training!")
    else:
        print("⚠️  Bug not clearly reproduced (may need more steps or different conditions)")
    print()
    
    # Plot comparison
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # CV comparison
        axes[0].plot(results['Correct']['cv'], label='Correct (EMA only in backward)', linewidth=2, color='green')
        axes[0].plot(results['Buggy']['cv'], label='Buggy (EMA in forward + backward)', linewidth=2, color='red')
        axes[0].axhline(0.005, color='blue', linestyle='--', label='Target 0.005')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('CV (Log Scale)')
        axes[0].set_title('CV Comparison: Forward EMA Duplication Bug')
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)
        
        # Lambda std comparison
        axes[1].plot(results['Correct']['lambda_std'], label='Correct Lambda Std', linewidth=2, color='green')
        axes[1].plot(results['Buggy']['lambda_std'], label='Buggy Lambda Std', linewidth=2, color='red')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Lambda Standard Deviation')
        axes[1].set_title('Dual Variable Stability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Expert load std comparison
        axes[2].plot(results['Correct']['expert_load_std'], label='Correct Load Std', linewidth=2, color='green')
        axes[2].plot(results['Buggy']['expert_load_std'], label='Buggy Load Std', linewidth=2, color='red')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Expert Load Standard Deviation')
        axes[2].set_title('Expert Load Distribution Stability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = project_root / 'forward_ema_duplication_bug.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Bug reproduction plot saved to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plot: {e}")
    
    return b_final_cv > c_final_cv * 1.5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--momentum", action="store_true", help="Use momentum filtering")
    parser.add_argument("--compare", action="store_true", help="Compare vanilla vs momentum")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--test-bug", action="store_true", help="Test forward EMA duplication bug")
    args = parser.parse_args()
    
    if args.test_bug:
        success = test_forward_ema_duplication_bug()
        sys.exit(0 if success else 1)
    elif args.compare:
        success = compare_vanilla_vs_momentum()
        sys.exit(0 if success else 1)
    else:
        success = verify_dual_primal_router(use_momentum=args.momentum, batch_size=args.batch_size)
        sys.exit(0 if success else 1)
