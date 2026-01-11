"""
HERO Distributed Training & Inference Module
分布式训练与推理模块

This module provides utilities for distributed training and inference:
1. DDP (Distributed Data Parallel) - Multi-GPU training
2. FSDP (Fully Sharded Data Parallel) - Memory-efficient large model training
3. DeepSpeed Integration - ZeRO optimization stages
4. Distributed Inference - Multi-GPU inference with model parallelism

Reference: HERO Engineering Optimization Requirements
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, List, Callable, Tuple
from contextlib import contextmanager
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Basic Distributed Setup
# ============================================================================

def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> Tuple[int, int, torch.device]:
    """
    Initialize distributed training environment.
    
    Supports both torchrun and manual initialization.
    
    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        init_method: URL for process group initialization.
        world_size: Total number of processes (auto-detected if None).
        rank: Process rank (auto-detected if None).
        
    Returns:
        Tuple of (rank, world_size, device).
    """
    # Check if already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        return rank, world_size, device
    
    # Auto-detect from environment (torchrun sets these)
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        backend = "gloo"
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, device={device}")
    else:
        logger.info(f"Running in single-process mode on {device}")
    
    return rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


@contextmanager
def main_process_first():
    """
    Context manager that ensures main process runs first.
    Useful for downloading/caching operations.
    """
    if not is_main_process():
        synchronize()
    yield
    if is_main_process():
        synchronize()


# ============================================================================
# 2. DDP Wrapper
# ============================================================================

def wrap_model_ddp(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: The model to wrap.
        device_ids: CUDA devices for the model.
        output_device: Device for output gathering.
        find_unused_parameters: Whether to find unused parameters (slower).
        gradient_as_bucket_view: Memory optimization.
        static_graph: Enable static graph optimization.
        
    Returns:
        DDP-wrapped model.
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning unwrapped model")
        return model
    
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    
    if device_ids is None:
        device_ids = [local_rank]
    if output_device is None:
        output_device = local_rank
    
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )
    
    logger.info(f"Wrapped model with DDP on device {device_ids}")
    return ddp_model


# ============================================================================
# 3. FSDP Support (Memory-Efficient Large Model Training)
# ============================================================================

def get_fsdp_config(
    model: nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    activation_checkpointing: bool = True,
    auto_wrap_policy: Optional[str] = "transformer_layer",
) -> Dict[str, Any]:
    """
    Get FSDP configuration for memory-efficient training.
    
    Args:
        model: The model to configure FSDP for.
        sharding_strategy: One of "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD".
        cpu_offload: Whether to offload parameters to CPU.
        mixed_precision: Whether to use mixed precision.
        activation_checkpointing: Whether to use activation checkpointing.
        auto_wrap_policy: Policy for auto-wrapping layers.
        
    Returns:
        FSDP configuration dictionary.
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            CPUOffload,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
        )
    except ImportError:
        raise ImportError("FSDP requires PyTorch >= 1.12")
    
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # CPU offload
    offload_config = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Mixed precision config
    mp_config = None
    if mixed_precision:
        mp_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # Auto wrap policy
    wrap_policy = None
    if auto_wrap_policy == "transformer_layer":
        # Find transformer layer classes in model
        layer_classes = set()
        for name, module in model.named_modules():
            if "TransformerEncoderLayer" in type(module).__name__:
                layer_classes.add(type(module))
            elif "DecoderLayer" in type(module).__name__:
                layer_classes.add(type(module))
        if layer_classes:
            wrap_policy = transformer_auto_wrap_policy(layer_classes)
    elif auto_wrap_policy == "size_based":
        wrap_policy = size_based_auto_wrap_policy(min_num_params=1e8)
    
    config = {
        "sharding_strategy": sharding,
        "cpu_offload": offload_config,
        "mixed_precision": mp_config,
        "auto_wrap_policy": wrap_policy,
        "use_orig_params": True,
        "limit_all_gathers": True,
    }
    
    logger.info(f"FSDP config: strategy={sharding_strategy}, cpu_offload={cpu_offload}, mp={mixed_precision}")
    return config


def wrap_model_fsdp(
    model: nn.Module,
    fsdp_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Wrap model with Fully Sharded Data Parallel.
    
    Args:
        model: The model to wrap.
        fsdp_config: Configuration from get_fsdp_config().
        
    Returns:
        FSDP-wrapped model.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        raise ImportError("FSDP requires PyTorch >= 1.12")
    
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning unwrapped model")
        return model
    
    if fsdp_config is None:
        fsdp_config = get_fsdp_config(model)
    
    fsdp_model = FSDP(model, **fsdp_config)
    logger.info("Wrapped model with FSDP")
    
    return fsdp_model


# ============================================================================
# 4. DeepSpeed Integration
# ============================================================================

def get_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    gradient_accumulation_steps: int = 1,
    train_micro_batch_size_per_gpu: int = 1,
    fp16_enabled: bool = False,
    bf16_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Generate DeepSpeed configuration.
    
    Args:
        stage: ZeRO optimization stage (0, 1, 2, or 3).
        offload_optimizer: Offload optimizer states to CPU.
        offload_param: Offload parameters to CPU (stage 3 only).
        gradient_accumulation_steps: Number of gradient accumulation steps.
        train_micro_batch_size_per_gpu: Batch size per GPU.
        fp16_enabled: Use FP16 mixed precision.
        bf16_enabled: Use BF16 mixed precision.
        
    Returns:
        DeepSpeed configuration dictionary.
    """
    config = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    
    # Offload configurations
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    if offload_param and stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    # Precision settings
    if bf16_enabled:
        config["bf16"] = {"enabled": True}
    elif fp16_enabled:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    
    logger.info(f"DeepSpeed config: ZeRO stage={stage}, bf16={bf16_enabled}, fp16={fp16_enabled}")
    return config


def initialize_deepspeed(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    model_parameters: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize model with DeepSpeed.
    
    Args:
        model: The model to wrap.
        optimizer: Optional optimizer (DeepSpeed can create one).
        config: DeepSpeed configuration.
        model_parameters: Model parameters for optimizer.
        
    Returns:
        Tuple of (model_engine, optimizer, _, lr_scheduler).
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError("DeepSpeed not installed. Install with: pip install deepspeed")
    
    if config is None:
        config = get_deepspeed_config()
    
    if model_parameters is None:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        model_parameters=model_parameters,
    )
    
    logger.info("Initialized DeepSpeed engine")
    return model_engine, optimizer, None, lr_scheduler


# ============================================================================
# 5. Distributed Inference
# ============================================================================

class DistributedInferenceEngine:
    """
    Distributed inference engine for multi-GPU inference.
    
    Supports:
    - Simple data parallelism (each GPU processes different samples)
    - Pipeline parallelism (model split across GPUs)
    - Tensor parallelism (layers split across GPUs)
    
    Args:
        model: The model to run inference on.
        mode: Inference mode ('data_parallel', 'pipeline', 'tensor').
    """
    
    def __init__(
        self,
        model: nn.Module,
        mode: str = "data_parallel",
    ):
        self.model = model
        self.mode = mode
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
    def distribute_batch(
        self, 
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Distribute a batch across GPUs.
        
        Args:
            batch: Dictionary of tensors.
            
        Returns:
            Local batch for this GPU.
        """
        if self.world_size == 1:
            return batch
        
        local_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch_size = tensor.size(0)
                per_gpu = batch_size // self.world_size
                start = self.rank * per_gpu
                end = start + per_gpu if self.rank < self.world_size - 1 else batch_size
                local_batch[key] = tensor[start:end]
            else:
                local_batch[key] = tensor
        
        return local_batch
    
    @torch.no_grad()
    def inference(
        self,
        batch: Dict[str, torch.Tensor],
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run distributed inference.
        
        Args:
            batch: Input batch.
            gather_outputs: Whether to gather outputs to rank 0.
            
        Returns:
            Model outputs (gathered if requested).
        """
        self.model.eval()
        
        # Distribute batch
        local_batch = self.distribute_batch(batch)
        
        # Run local inference
        local_outputs = self.model(**local_batch)
        
        # Gather outputs
        if gather_outputs and self.world_size > 1:
            gathered_outputs = self._gather_outputs(local_outputs)
            return gathered_outputs
        
        return local_outputs
    
    def _gather_outputs(
        self, 
        outputs: Any,
    ) -> Any:
        """Gather outputs from all GPUs to rank 0."""
        if isinstance(outputs, torch.Tensor):
            gathered = [torch.zeros_like(outputs) for _ in range(self.world_size)]
            dist.all_gather(gathered, outputs)
            return torch.cat(gathered, dim=0)
        elif isinstance(outputs, dict):
            return {k: self._gather_outputs(v) for k, v in outputs.items()}
        elif hasattr(outputs, '_fields'):  # NamedTuple
            return type(outputs)(*[self._gather_outputs(f) for f in outputs])
        else:
            return outputs


# ============================================================================
# 6. Utility Functions
# ============================================================================

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce and average tensor across all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast Python object from source rank to all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj
    
    object_list = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def save_on_main_only(state_dict: Dict, path: str):
    """Save checkpoint only on main process."""
    if is_main_process():
        torch.save(state_dict, path)
        logger.info(f"Saved checkpoint to {path}")
    synchronize()


def get_distributed_sampler(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.distributed.DistributedSampler:
    """Get distributed sampler for dataset."""
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=shuffle,
        seed=seed,
    )
