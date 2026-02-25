#!/usr/bin/env python3
"""
Custom Optimizers for DeepSpeed Training
"""
import math
import torch
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from typing import Dict, Any, Optional


class LionOptimizer(DeepSpeedCPUAdam):
    """
    Lion Optimizer for DeepSpeed
    Based on: https://arxiv.org/abs/2302.06675
    """
    
    def __init__(self, model_params, lr=1e-4, weight_decay=0.01, beta1=0.9, beta2=0.99, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
        # Filter kwargs to only include what DeepSpeedCPUAdam expects
        valid_keys = {'bias_correction', 'amsgrad', 'adamw_mode', 'fp32_optimizer_states'}
        ds_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        # DeepSpeed CPU Adam을 기반으로 초기화
        super().__init__(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **ds_kwargs
        )

    def __del__(self):
        """Safe cleanup for DeepSpeedCPUAdam"""
        if hasattr(self, 'ds_opt_adam') and self.ds_opt_adam is not None:
            try:
                self.ds_opt_adam.destroy_adam(self.opt_id)
            except Exception:
                pass
    
    def step(self, closure=None):
        """Custom Lion optimizer step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = self.beta1, self.beta2
                
                state['step'] += 1
                
                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                update = torch.sign(update)
                
                # Weight decay
                if self.weight_decay != 0:
                    p.data.add_(p.data, alpha=-self.weight_decay * group['lr'])
                
                # Parameter update
                p.data.add_(update, alpha=-group['lr'])
                
                # Update moving averages
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class AdaFactorOptimizer(DeepSpeedCPUAdam):
    """
    AdaFactor Optimizer for DeepSpeed
    Memory-efficient optimizer for large models
    """
    
    def __init__(self, model_params, lr=1e-3, weight_decay=0.01, 
                 beta1=0.9, beta2=0.999, eps1=1e-30, eps2=1e-3, 
                 cliping_threshold=1.0, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.cliping_threshold = cliping_threshold
        self.weight_decay = weight_decay
        
        # Filter kwargs to only include what DeepSpeedCPUAdam expects
        valid_keys = {'bias_correction', 'amsgrad', 'adamw_mode', 'fp32_optimizer_states'}
        ds_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        super().__init__(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps1,
            **ds_kwargs
        )

    def __del__(self):
        """Safe cleanup for DeepSpeedCPUAdam"""
        if hasattr(self, 'ds_opt_adam') and self.ds_opt_adam is not None:
            try:
                self.ds_opt_adam.destroy_adam(self.opt_id)
            except Exception:
                pass
    
    def step(self, closure=None):
        """Custom AdaFactor optimizer step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_row'] = torch.zeros(grad.size(0), device=grad.device)
                    state['exp_avg_sq_col'] = torch.zeros(grad.size(1), device=grad.device)
                
                exp_avg = state['exp_avg']
                exp_avg_sq_row = state['exp_avg_sq_row']
                exp_avg_sq_col = state['exp_avg_sq_col']
                beta1, beta2 = self.beta1, self.beta2
                
                state['step'] += 1
                
                # AdaFactor update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update row and column moving averages
                exp_avg_sq_row.mul_(beta2).add_(grad.pow(2).mean(dim=1), alpha=1 - beta2)
                exp_avg_sq_col.mul_(beta2).add_(grad.pow(2).mean(dim=0), alpha=1 - beta2)
                
                # Compute RMS
                rms_row = exp_avg_sq_row.sqrt().add_(self.eps1)
                rms_col = exp_avg_sq_col.sqrt().add_(self.eps1)
                
                # Compute update
                update = exp_avg / (rms_row.unsqueeze(1) * rms_col.unsqueeze(0) + self.eps2)
                
                # Gradient clipping
                if self.cliping_threshold > 0:
                    update = torch.clamp(update, -self.cliping_threshold, self.cliping_threshold)
                
                # Weight decay
                if self.weight_decay != 0:
                    p.data.add_(p.data, alpha=-self.weight_decay * group['lr'])
                
                # Parameter update
                p.data.add_(update, alpha=-group['lr'])
        
        return loss


class SophiaOptimizer(DeepSpeedCPUAdam):
    """
    Sophia Optimizer for DeepSpeed
    Based on: https://arxiv.org/abs/2305.14342
    """
    
    def __init__(self, model_params, lr=1e-4, weight_decay=0.01, 
                 beta1=0.965, beta2=0.99, rho=0.01, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.weight_decay = weight_decay
        
        # Filter kwargs to only include what DeepSpeedCPUAdam expects
        valid_keys = {'bias_correction', 'amsgrad', 'adamw_mode', 'fp32_optimizer_states'}
        ds_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        super().__init__(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            **ds_kwargs
        )

    def __del__(self):
        """Safe cleanup for DeepSpeedCPUAdam"""
        if hasattr(self, 'ds_opt_adam') and self.ds_opt_adam is not None:
            try:
                self.ds_opt_adam.destroy_adam(self.opt_id)
            except Exception:
                pass
    
    def step(self, closure=None):
        """Custom Sophia optimizer step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                hessian = state['hessian']
                beta1, beta2 = self.beta1, self.beta2
                
                state['step'] += 1
                
                # Update moving average of gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Estimate Hessian (simplified version)
                if state['step'] % 10 == 0:  # Update Hessian every 10 steps
                    hessian.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                # Sophia update
                update = exp_avg / (hessian.sqrt() + self.rho)
                
                # Weight decay
                if self.weight_decay != 0:
                    p.data.add_(p.data, alpha=-self.weight_decay * group['lr'])
                
                # Parameter update
                p.data.add_(update, alpha=-group['lr'])
        
        return loss


class MuonOptimizer(DeepSpeedCPUAdam):
    """
    Muon Optimizer for DeepSpeed
    Based on: https://arxiv.org/abs/2502.16982
    """
    def __init__(
        self,
        model_params,
        lr=5e-5,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        **kwargs
    ):
        # Filter kwargs to only include what DeepSpeedCPUAdam expects
        valid_keys = {'bias_correction', 'amsgrad', 'adamw_mode', 'fp32_optimizer_states'}
        ds_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        # DeepSpeed CPU Adam을 기반으로 초기화
        super().__init__(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=adamw_betas,
            eps=adamw_eps,
            **ds_kwargs
        )

    def __del__(self):
        """Safe cleanup for DeepSpeedCPUAdam"""
        if hasattr(self, 'ds_opt_adam') and self.ds_opt_adam is not None:
            try:
                self.ds_opt_adam.destroy_adam(self.opt_id)
            except Exception:
                pass
        
        # Muon-specific parameters
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adamw_betas = adamw_betas
        self.adamw_eps = adamw_eps
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group['params']:
                # Use Muon for 2D parameters that don't look like embeddings or heads
                if p.ndim == 2 and "embed" not in str(p) and "lm_head" not in str(p):
                    self.state[p]["use_muon"] = True
                else:
                    self.state[p]["use_muon"] = False

    def zeropower_via_newtonschulz5(self, G, steps):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(0) > G.size(1):
            X = X.T
        # Ensure spectral norm is at most 1
        X = X / (X.norm() + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.T
            B = (
                b * A + c * A @ A
            )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X

        if G.size(0) > G.size(1):
            X = X.T
        return X


    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(self.momentum).add_(g)
                if self.nesterov:
                    g = g.add(buf, alpha=self.momentum)
                else:
                    g = buf
                u = self.zeropower_via_newtonschulz5(g, steps=self.ns_steps)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = self.adamw_betas
            eps = self.adamw_eps
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_custom_optimizer(optimizer_name: str, model_params, **kwargs):
    """
    Factory function to get custom optimizer
    
    Args:
        optimizer_name: Name of the optimizer ('lion', 'adafactor', 'sophia')
        model_params: Model parameters
        **kwargs: Optimizer-specific parameters
    
    Returns:
        Custom optimizer instance
    """
    optimizer_map = {
        'lion': LionOptimizer,
        'adafactor': AdaFactorOptimizer,
        'sophia': SophiaOptimizer,
        'muon': MuonOptimizer,
    }
    
    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizer_map.keys())}")
    
    optimizer_class = optimizer_map[optimizer_name.lower()]
    return optimizer_class(model_params, **kwargs)
    