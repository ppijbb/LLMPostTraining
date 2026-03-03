import os
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from PIL import Image

def setup_dataset(data_config: Dict[str, Any], tokenizer, logger: logging.Logger, training_config: Dict[str, Any] = None, allow_text_only: bool = False) -> Tuple[Dict, Callable]:
    """
    Qwen3-VL-MoE 전용: Native Processor를 100% 활용하는 데이터셋 설정.
    Universal Exoskeleton 구조를 위해 불필요한 래퍼를 제거함.
    """
    from data.multi_domain_sft_dataset import get_multi_domain_sft_dataset, DOMAIN_TO_ID
    
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 100000)
    max_seq_length = data_config.get("max_seq_length", 131072)
    test_size = data_config.get("test_size", 0.005)
    use_streaming = data_config.get("streaming", False)
    
    logger.info(f"💾 Loading Multi-Modal Dataset: {dataset_name} (Max: {max_samples})")
    
    # 데이터셋 로드 (텍스트/이미지 포함)
    dataset = get_multi_domain_sft_dataset(
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_samples_per_domain=max_samples,
        test_size=test_size,
        use_streaming=use_streaming,
        allow_text_only=False # 반드시 멀티모달 사용
    )

    # -------------------------------------------------------------
    # [FIX] Native Processor 기반의 가장 단순한 Collator
    # Qwen3-VL은 반드시 processor를 통해 text와 image를 한꺼번에 tokenize해야 함.
    #
    # [ZeRO-3 Deadlock 방지] 모든 샘플이 동일한 computation path를 따라야 함.
    # - 텍스트 전용(이미지 없음)이라도: 이미지 placeholder + dummy 이미지 주입
    # - 모든 rank가 항상 (pixel_values 포함) 동일한 VLM forward/backward를 수행
    # -------------------------------------------------------------

    def pure_qwen_collate_fn(examples):
        # examples: List[Dict["messages": ..., "images": ...]]
        texts = []
        images = []
        for example in examples:
            msgs = example["messages"]
            has_image_placeholder = any(
                any(c.get("type") == "image" for c in m.get("content", []))
                for m in msgs if isinstance(m.get("content"), list)
            )
            has_actual_images = bool(example.get("images"))

            # [ZeRO-3] 이미지 placeholder가 없으면 반드시 추가 (텍스트 전용 → 동일 path 강제)
            if not has_image_placeholder:
                for m in msgs:
                    if m["role"] == "user":
                        if isinstance(m["content"], str):
                            m["content"] = [{"type": "image"}, {"type": "text", "text": m["content"]}]
                        elif isinstance(m["content"], list):
                            m["content"].insert(0, {"type": "image"})
                        break
                has_image_placeholder = True

            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)

            # [ZeRO-3] 항상 이미지 1개 per sample. placeholder와 개수 일치 필수.
            if has_actual_images:
                img = example["images"][0] if isinstance(example["images"], list) else example["images"]
                images.append(img)
            else:
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        # [0vs2048] rank별 vision patch 수 불일치(256 vs 1064) 방지: 모든 이미지를 동일 해상도로 맞춤 → processor 출력 patch 수 동일.
        _fix_size = (224, 224)
        def _resize_to_fixed(img):
            if not hasattr(img, "resize"):
                return img
            if hasattr(img, "size") and img.size != _fix_size:
                if getattr(img, "mode", "RGB") != "RGB":
                    img = img.convert("RGB")
                return img.resize(_fix_size, Image.BILINEAR)
            return img
        images = [_resize_to_fixed(im) for im in images]

        # [ZeRO-3] 모든 샘플에 이미지 있음 → 항상 동일 path. None/혼합 분기 제거.
        batch = tokenizer(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        )
        
        # Labels 생성
        batch["labels"] = batch["input_ids"].clone()
        # Padding (-100) 및 Vision Special Tokens 마스킹
        # Qwen3-VL-MoE의 pad_token_id 확인
        pad_id = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.pad_token_id
        if pad_id is not None:
            batch["labels"][batch["labels"] == pad_id] = -100
            
        # [CRITICAL] Vision Token ID들(-100 마스킹)을 하드코딩하지 않고 동적으로 처리
        # Qwen3-VL-MoE: <|vision_start|>, <|image_pad|>, <|vision_end|> 등
        # 이들은 학습 대상이 아님
        for token_name in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]:
            try:
                tid = tokenizer.tokenizer.convert_tokens_to_ids(token_name)
                batch["labels"][batch["labels"] == tid] = -100
            except Exception:
                pass

        # [0vs2048] 디버깅 시에만: 첫 배치 1회 로그 (SEQORTH_DEBUG_0VS2048=1 일 때만)
        if os.environ.get("SEQORTH_DEBUG_0VS2048") == "1" and not getattr(pure_qwen_collate_fn, "_logged_once", False):
            pure_qwen_collate_fn._logged_once = True
            _pv = batch.get("pixel_values")
            _grid = batch.get("image_grid_thw")
            _num_patches = int(_pv.shape[0]) if _pv is not None and hasattr(_pv, "shape") else None
            _grid_str = tuple(_grid.shape) if _grid is not None and hasattr(_grid, "shape") else str(_grid)
            print(f"[0vs2048] COLLATOR CHECK: first batch | pixel_values.shape={getattr(_pv, 'shape', None)}, image_grid_thw.shape={_grid_str} | num_patches(seq)={_num_patches} (must be same on all ranks)", flush=True)

        # Domain ids for routing metrics (MoE callback)
        domain_ids_list = [DOMAIN_TO_ID.get(example.get("domain") or "", -1) for example in examples]
        batch["domain_ids"] = torch.tensor(domain_ids_list, dtype=torch.long)

        return batch

    if os.environ.get("SEQORTH_DEBUG_0VS2048") == "1":
        logger.info("[0vs2048] Collator: pure_qwen_collate_fn | image resize 224x224 → uniform vision patch count across ranks")
    return dataset, pure_qwen_collate_fn
