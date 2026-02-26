#!/usr/bin/env python3
"""
Vision path만 DeepSpeed ZeRO-2 + 다중 GPU로 테스트.
학습과 동일 조건: 동일 데이터셋·collator, ZeRO-2, 분산 backward로 0 vs 2048/4608 재현.
실행: accelerate launch로 (run_test_vision.sh 사용 권장)
"""
import os
import sys
import json
import logging
import torch
import torch.distributed as dist

# 프로젝트 루트
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Expect PYTHONPATH to include repo root

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 분산 초기화 (train과 동일)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config_path = os.path.join(ROOT, "config", "seqorth", "seqorth_qwen_config.json")
    if not os.path.isfile(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    data = load_config(config_path)
    model_config = data["model_config"]
    data_config = data["data_config"]
    training_config = data.get("training_config", {})

    # 1) 패치 먼저 적용 (train과 동일)
    from training.seqorth_sft.train_seqorth import _patch_qwen3_participation_guard
    _patch_qwen3_participation_guard()

    # 2) 토크나이저
    from training.seqorth_sft.train_seqorth import setup_tokenizer
    if rank == 0:
        logger.info("Loading tokenizer...")
    tokenizer = setup_tokenizer(model_config)

    # 3) 데이터셋 + collator, DistributedSampler로 rank별 배치 (train과 동일)
    from core.training_utils.dataset_utils import setup_dataset
    if rank == 0:
        logger.info("Loading dataset and collator...")
    dataset, collate_fn = setup_dataset(data_config, tokenizer, logger, training_config)
    train_ds = dataset.get("train") or dataset.get("train_dataset") or list(dataset.values())[0]
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=dist.get_world_size(), rank=rank, shuffle=False
    )
    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    image_grid_thw = batch["image_grid_thw"].to(device)

    # [검증] rank별 이미지 유무: ds 마다 텍스트만 / 이미지 있음 차이로 0 vs 2048 발생하는지 확인
    from transformers import AutoConfig
    model_name = model_config.get("model_name_or_path", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    image_token_id = getattr(config, "image_token_id", None)
    num_patches = int(pixel_values.shape[0]) if pixel_values.dim() > 0 else 0
    num_image_tokens = 0
    if "input_ids" in batch and image_token_id is not None:
        num_image_tokens = (batch["input_ids"] == image_token_id).sum().item()
    # rank마다 로그 한 줄씩 (순서 보장 위해 barrier)
    dist.barrier()
    _ids = batch.get("input_ids")
    _ids_shape = tuple(_ids.shape) if _ids is not None and hasattr(_ids, "shape") else None
    print(
        f"[vision_test] rank={rank} num_patches={num_patches} num_image_tokens_in_input_ids={num_image_tokens} "
        f"pixel_values.shape={tuple(pixel_values.shape)} input_ids.shape={_ids_shape}",
        flush=True,
    )
    dist.barrier()
    # 모든 rank가 동일한 num_image_tokens / num_patches 여야 full train에서 0 vs 2048 안 남
    if rank == 0:
        logger.info("If any rank has num_image_tokens=0 or num_patches=0 → that rank gets 0 gradient to vision → 0 vs 2048 in full train.")

    # 4) 전체 모델 CPU 로드 후 .visual 만 추출
    from transformers import AutoModelForImageTextToText
    model_name = model_config.get("model_name_or_path", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    if rank == 0:
        logger.info(f"Loading model from {model_name} (CPU), using .visual only...")
    full = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    visual = full.model.visual
    del full
    torch.cuda.empty_cache()
    visual = visual.to(device=device, dtype=torch.bfloat16)
    visual.train()

    # 5) DeepSpeed ZeRO-2로 감싸기 (학습과 동일 backward 경로)
    ds_config_path = os.path.join(ROOT, "config", "seqorth", "deepspeed_zero2.json")
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    # 테스트용: "auto" 제거, 전부 숫자로 (scheduler/optimizer가 str 연산하면 TypeError)
    world_size = dist.get_world_size()
    ds_config["optimizer"] = {"type": "AdamW", "params": {"lr": 1e-5, "weight_decay": 0.01}}
    ds_config["train_batch_size"] = world_size
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["gradient_clipping"] = 1.0
    ds_config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 0,
            "total_num_steps": 1,
            "last_batch_iteration": -1,
        },
    }

    import deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=visual,
        model_parameters=visual.parameters(),
        config=ds_config,
    )
    if rank == 0:
        logger.info("DeepSpeed ZeRO-2 engine initialized (vision only).")

    # 6) Vision forward + DeepSpeed backward (학습과 동일)
    pixel_values = pixel_values.to(dtype=model_engine.module.dtype)
    image_embeds, deepstack_image_embeds = model_engine(pixel_values, grid_thw=image_grid_thw)
    if rank == 0:
        logger.info(f"Vision out: image_embeds={image_embeds.shape}, deepstack len={len(deepstack_image_embeds)}")

    loss = image_embeds.sum()
    if deepstack_image_embeds:
        loss = loss + sum(e.sum() for e in deepstack_image_embeds)
    if rank == 0:
        logger.info("Running DeepSpeed backward...")
    model_engine.backward(loss)
    model_engine.step()
    if rank == 0:
        logger.info("OK: Vision backward (ZeRO-2) completed without error.")


if __name__ == "__main__":
    main()
