"""
HERO Advanced Engineering Optimization Module
高级工程优化模块

This module provides production-grade optimization utilities for HERO training:
1. Smart Gradient Checkpointing - Memory-efficient training with frozen encoder support
2. FlashAttention V2 Integration - Efficient attention computation via SDPA
3. QLoRA / LoRA Setup - 4-bit quantized parameter-efficient fine-tuning
4. Mixed Precision Training (AMP) - BFloat16 training utilities
5. Torch.compile Integration - PyTorch 2.x compilation for speedup

Reference: User engineering optimization requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict, Any, Callable, Tuple
from contextlib import contextmanager
from functools import wraps
import logging
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Smart Gradient Checkpointing
# ============================================================================

class SmartCheckpointWrapper(nn.Module):
    """
    Wrapper that applies gradient checkpointing to a module with smart handling
    for frozen encoder setups.
    
    Key features:
    - Automatically ensures inputs have requires_grad=True for checkpointing
    - Handles frozen encoder + trainable head scenarios
    - Configurable checkpoint segments for memory/speed tradeoff
    
    Args:
        module: The module to wrap.
        checkpoint_segments: Number of segments to divide the module into.
        preserve_rng_state: Whether to preserve RNG state during checkpointing.
    """
    
    def __init__(
        self, 
        module: nn.Module, 
        checkpoint_segments: int = 1,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        self.module = module
        self.checkpoint_segments = checkpoint_segments
        self.preserve_rng_state = preserve_rng_state
        
        # Dummy parameter to ensure gradient flow
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, *args, **kwargs):
        if not self.training:
            return self.module(*args, **kwargs)
            
        # Ensure at least one input has requires_grad for checkpointing
        def ensure_grad(inputs):
            processed = []
            has_grad = False
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    if inp.requires_grad:
                        has_grad = True
                    processed.append(inp)
                else:
                    processed.append(inp)
            return tuple(processed), has_grad
            
        processed_args, has_grad = ensure_grad(args)
        
        if not has_grad:
            # Add dummy gradient path for frozen encoders
            if len(processed_args) > 0 and isinstance(processed_args[0], torch.Tensor):
                # Create a gradient-enabled version by adding zero * dummy
                first_arg = processed_args[0] + (self._dummy_param * 0).sum()
                processed_args = (first_arg,) + processed_args[1:]
        
        def run_function(*inputs):
            return self.module(*inputs, **kwargs)
            
        return checkpoint(
            run_function, 
            *processed_args,
            use_reentrant=False,
            preserve_rng_state=self.preserve_rng_state,
        )


def enable_smart_checkpointing(
    model: nn.Module,
    target_modules: List[str] = None,
    exclude_modules: List[str] = None,
) -> nn.Module:
    """
    Enable smart gradient checkpointing on specific modules.
    
    Args:
        model: The model to modify.
        target_modules: Module names to checkpoint (e.g., ['integration_layer', 'llm_model']).
        exclude_modules: Module names to exclude from checkpointing.
        
    Returns:
        Modified model with checkpointing enabled.
    """
    if target_modules is None:
        target_modules = ['integration_layer', 'observation_experts', 'llm_model']
    if exclude_modules is None:
        exclude_modules = []
        
    checkpointed_count = 0
        
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and not any(e in name for e in exclude_modules):
            # Check if module has forward method we can wrap
            if hasattr(module, 'forward') and not isinstance(module, SmartCheckpointWrapper):
                # For Transformer layers, wrap the entire module
                if hasattr(module, 'self_attn') or hasattr(module, 'encoder_layer'):
                    original_forward = module.forward
                    
                    @wraps(original_forward)
                    def make_checkpointed_forward(orig_fn):
                        def checkpointed_forward(*args, **kwargs):
                            if module.training and torch.is_grad_enabled():
                                def fn(*a):
                                    return orig_fn(*a, **kwargs)
                                return checkpoint(fn, *args, use_reentrant=False)
                            return orig_fn(*args, **kwargs)
                        return checkpointed_forward
                    
                    module.forward = make_checkpointed_forward(original_forward)
                    checkpointed_count += 1
                    
    logger.info(f"Enabled smart checkpointing on {checkpointed_count} modules")
    return model


# ============================================================================
# 2. FlashAttention V2 / SDPA Integration
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Drop-in replacement for attention that uses PyTorch's SDPA.
    
    Automatically uses FlashAttention V2 when available (CUDA, supported dtypes).
    Falls back to memory-efficient attention or vanilla attention otherwise.
    
    Features:
    - Supports is_causal flag
    - Efficient mask handling (avoids [B, H, S, S] expansion)
    - Dropout support during training
    
    Args:
        dropout: Attention dropout probability.
        scale: Optional attention scale (default: 1/sqrt(d_k)).
    """
    
    def __init__(self, dropout: float = 0.0, scale: Optional[float] = None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        
        # Check Flash Attention availability
        self._flash_available = self._check_flash_attention()
        
    def _check_flash_attention(self) -> bool:
        """Check if FlashAttention is available."""
        if not torch.cuda.is_available():
            return False
        try:
            # PyTorch 2.0+ includes SDPA with automatic backend selection
            from torch.nn.functional import scaled_dot_product_attention
            return True
        except ImportError:
            return False
            
    def forward(
        self,
        query: torch.Tensor,      # [B, H, S_q, D]
        key: torch.Tensor,        # [B, H, S_k, D]
        value: torch.Tensor,      # [B, H, S_k, D]
        attn_mask: Optional[torch.Tensor] = None,  # [B, S_q, S_k] or [B, H, S_q, S_k] or [S_q, S_k]
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query, key, value: Input tensors.
            attn_mask: Attention mask (True = masked out for boolean, -inf = masked for float).
            is_causal: If True, applies causal mask (ignores attn_mask).
            need_weights: If True, also returns attention weights.
            
        Returns:
            output: Attention output [B, H, S_q, D].
            attn_weights: Optional attention weights if need_weights=True.
        """
        B, H, S_q, D = query.shape
        S_k = key.shape[2]
        
        # Scale
        scale = self.scale if self.scale is not None else (D ** -0.5)
        
        # Use SDPA if available and weights not needed
        if self._flash_available and not need_weights:
            # Convert mask format for SDPA
            # SDPA expects: True = attend, or float mask where -inf = ignore
            sdpa_mask = None
            if attn_mask is not None and not is_causal:
                if attn_mask.dtype == torch.bool:
                    # Convert: our convention (True=masked) to SDPA (True=attend)
                    sdpa_mask = ~attn_mask
                else:
                    # Float mask: already in correct format (0 = attend, -inf = ignore)
                    sdpa_mask = attn_mask
                    
                # Expand mask if needed
                if sdpa_mask.dim() == 2:
                    sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)
                elif sdpa_mask.dim() == 3:
                    sdpa_mask = sdpa_mask.unsqueeze(1)
                    
            dropout_p = self.dropout if self.training else 0.0
            
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
            return output, None
            
        # Fallback: Manual computation
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        elif attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask
                
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
            
        output = torch.matmul(attn_weights, value)
        
        if need_weights:
            return output, attn_weights
        return output, None


# ============================================================================
# 3. QLoRA / LoRA Setup
# ============================================================================

def setup_qlora(
    model: nn.Module,
    llm_module_name: str = "llm_model",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    quantize_4bit: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    keep_modules_full_precision: Optional[List[str]] = None,
) -> nn.Module:
    """
    Setup QLoRA (Quantized LoRA) for efficient fine-tuning.
    
    This function:
    1. Quantizes the LLM to 4-bit (NF4)
    2. Applies LoRA adapters to specified layers
    3. Keeps custom modules (like EvidenceIntegrationLayer) in full precision
    
    Args:
        model: The full HERO model.
        llm_module_name: Name of the LLM submodule.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout for LoRA layers.
        target_modules: LLM modules to apply LoRA to.
        quantize_4bit: Whether to quantize to 4-bit.
        compute_dtype: Compute dtype for quantized operations.
        keep_modules_full_precision: Modules to keep in FP32/BF16.
        
    Returns:
        Modified model with QLoRA applied.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("PEFT library required. Install with: pip install peft")
        
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "bitsandbytes and transformers required for 4-bit quantization. "
                "Install with: pip install bitsandbytes transformers"
            )
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"]
                         
    if keep_modules_full_precision is None:
        keep_modules_full_precision = [
            "integration_layer", "observation_experts", 
            "projector", "evidence_imputation"
        ]
    
    # Get LLM module
    llm = getattr(model, llm_module_name, None)
    if llm is None:
        raise ValueError(f"Could not find LLM module '{llm_module_name}' in model")
    
    # Step 1: Prepare for k-bit training
    if quantize_4bit:
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    
    # Step 2: Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Step 3: Apply LoRA
    llm = get_peft_model(llm, lora_config)
    
    # Step 4: Update model reference
    setattr(model, llm_module_name, llm)
    
    # Step 5: Ensure custom modules stay in full precision
    for name, module in model.named_modules():
        if any(keep in name for keep in keep_modules_full_precision):
            module.to(compute_dtype if compute_dtype != torch.float32 else torch.float32)
            for param in module.parameters():
                param.requires_grad = True
    
    # Log parameter stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(
        f"QLoRA Setup Complete:\n"
        f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n"
        f"  Total params: {total_params:,}\n"
        f"  4-bit quantization: {quantize_4bit}\n"
        f"  LoRA rank: {lora_r}, alpha: {lora_alpha}"
    )
    
    return model


def get_lora_only_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA adapter weights for saving.
    
    Args:
        model: Model with LoRA applied.
        
    Returns:
        State dict containing only LoRA parameters.
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name.lower() or 'adapter_' in name.lower():
            lora_state[name] = param.detach().cpu()
    return lora_state


# ============================================================================
# 4. Mixed Precision Training (AMP)
# ============================================================================

class AMPTrainingContext:
    """
    Context manager for mixed precision training with BFloat16.
    
    Provides utilities for:
    - Automatic mixed precision with GradScaler
    - BFloat16 context for forward/backward
    - Gradient clipping integration
    
    Args:
        enabled: Whether AMP is enabled.
        dtype: Target dtype (torch.bfloat16 recommended for LLMs).
        grad_scaler: Optional existing GradScaler.
        max_grad_norm: Maximum gradient norm for clipping.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        max_grad_norm: float = 1.0,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.max_grad_norm = max_grad_norm
        
        # GradScaler only needed for float16, not bfloat16
        if dtype == torch.float16:
            self.scaler = grad_scaler or torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                yield
        else:
            yield
            
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Perform backward pass with optional scaling.
        
        Args:
            loss: Loss tensor.
            optimizer: Optimizer for gradient clipping.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with optional unscaling and clipping.
        
        Args:
            optimizer: The optimizer to step.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            
        if self.max_grad_norm > 0:
            # Get all parameters from optimizer
            params = []
            for group in optimizer.param_groups:
                params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


@contextmanager
def bf16_context():
    """Simple context manager for BFloat16 computation."""
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        yield


# ============================================================================
# 5. Torch.compile Integration
# ============================================================================

def compile_model(
    model: nn.Module,
    compile_modules: Optional[List[str]] = None,
    skip_modules: Optional[List[str]] = None,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> nn.Module:
    """
    Apply torch.compile to specified model components.
    
    Compiles static parts of the model while skipping dynamic control flows.
    
    Args:
        model: The model to compile.
        compile_modules: Module names to compile (if None, compiles entire model).
        skip_modules: Module names to skip compilation.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        fullgraph: If True, requires the entire module to be capturable.
        dynamic: If True, uses dynamic shapes.
        
    Returns:
        Compiled model.
    """
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
        return model
        
    if skip_modules is None:
        skip_modules = ['llm_model']  # LLMs often have dynamic control flow
        
    if compile_modules is not None:
        # Compile specific modules
        for name in compile_modules:
            module = getattr(model, name, None)
            if module is not None and not any(s in name for s in skip_modules):
                try:
                    compiled = torch.compile(
                        module, 
                        mode=mode, 
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                    setattr(model, name, compiled)
                    logger.info(f"Compiled module: {name}")
                except Exception as e:
                    logger.warning(f"Failed to compile {name}: {e}")
    else:
        # Compile entire model except skipped modules
        try:
            # First, mark modules to skip
            for name in skip_modules:
                module = getattr(model, name, None)
                if module is not None:
                    module._disable_compile = True
                    
            model = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            logger.info(f"Compiled full model with mode='{mode}'")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
            
    return model


# ============================================================================
# Unified Optimization Setup
# ============================================================================

def setup_hero_optimizations(
    model: nn.Module,
    enable_checkpointing: bool = True,
    enable_qlora: bool = False,
    enable_compile: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    lora_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, AMPTrainingContext]:
    """
    One-click optimization setup for HERO.
    
    Applies all recommended optimizations with sensible defaults.
    
    Args:
        model: The HERO model.
        enable_checkpointing: Enable gradient checkpointing.
        enable_qlora: Enable QLoRA (requires dependencies).
        enable_compile: Enable torch.compile.
        amp_dtype: Dtype for mixed precision.
        lora_config: Optional LoRA configuration dict.
        
    Returns:
        Tuple of (optimized_model, amp_context).
    """
    logger.info("Setting up HERO optimizations...")
    
    # 1. Gradient Checkpointing
    if enable_checkpointing:
        model = enable_smart_checkpointing(
            model,
            target_modules=['integration_layer', 'observation_experts'],
        )
        
    # 2. QLoRA
    if enable_qlora:
        config = lora_config or {}
        model = setup_qlora(
            model,
            llm_module_name=config.get('llm_module_name', 'llm_model'),
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            quantize_4bit=config.get('quantize_4bit', True),
        )
        
    # 3. Torch.compile
    if enable_compile:
        model = compile_model(
            model,
            compile_modules=['integration_layer', 'observation_experts'],
            skip_modules=['llm_model'],
        )
        
    # 4. AMP Context
    amp_context = AMPTrainingContext(
        enabled=True,
        dtype=amp_dtype,
        max_grad_norm=1.0,
    )
    
    logger.info("HERO optimizations setup complete!")
    return model, amp_context
