"""
Unsloth GRPO (Group Relative Policy Optimization) Trainer

This module provides GRPO training functionality using TRL's GRPOTrainer and Unsloth optimizations.
"""

import logging
import torch
import numpy as np
import json
import os
import copy
from typing import Dict, Any, List, Optional, Union
import traceback
# Import TRL components
from trl import GRPOTrainer, GRPOConfig

# Import Unsloth for model loading
from unsloth import FastLanguageModel

# Import transformers for callbacks
from transformers import TrainerCallback, TrainingArguments

# Import custom reward functions for TRL compatibility
from reward.reward_functions import MultiRewardFunction, SingleCustomRewardFunction
from reward.cmd_reward_functions import CommandRewardFunction, ComponentRewardWrapper

logger = logging.getLogger(__name__)


class GenerationLoggingCallback(TrainerCallback):
    """특정 step마다 생성된 텍스트를 콘솔과 파일에 출력하는 콜백 (항상 활성화)"""

    def __init__(
        self, 
        trainer, 
        output_dir: str = "./generation_logs", 
        max_samples: int = 5, 
        log_every_n_steps: int = 3
    ):
        self.trainer = trainer
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.log_every_n_steps = log_every_n_steps
        self.eval_step_count = 0

        # 로그 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"📊 Generation logging callback initialized. Output dir: {output_dir}, Log every {log_every_n_steps} steps")
        print(f"\n{'='*80}")
        print(f"📊 Generation Logging Callback 초기화됨")
        print(f"   출력 디렉토리: {output_dir}")
        print(f"   로깅 주기: 매 {log_every_n_steps} step마다")
        print(f"   최대 샘플 수: {max_samples}")
        print(f"{'='*80}\n")

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        """Step 종료 시 호출 (특정 step마다 반드시 실행)"""
        # 로깅 주기에 맞춰 생성 로그 출력 (무조건 실행)
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            self._log_generations(args, state, **kwargs)

    @torch.no_grad()
    def _log_generations(self, args: TrainingArguments, state, **kwargs):
        """실제 생성 로그 작성 (콘솔에 강제 출력)"""
        model = self.trainer.model
        tokenizer = self.trainer.tokenizer

        if not model or not tokenizer:
            logger.warning(f"⚠️ Step {state.global_step}: Model or tokenizer not available. Skipping generation logging.")
            return

        self.eval_step_count += 1
        current_step = state.global_step
        
        print(f"\n{'='*80}")
        print(f"🔄 STEP {current_step} - Generation 테스트 시작 (배치 처리)")
        print(f"{'='*80}")

        was_training = model.training
        model.eval()

        generation_logs = []
        sample_prompts = [
            "Start with 3, 5 4 6, mesial bleeding, middle of suppuration, mobility 2, 3 3 3, 3 2 3, repeat 8, 3 4 4, 3 5 4 furcation grade 2 ",
            "3 2 3, 3 2 3, 3 2 3, repeat, repeat, repeat, bleeding 1 on mesial, bleeding 2 mesial and distal three bleeding all",
            "number 1, 16 impacted, 17 and 32 missing, 5 4 3, 3 3 4, 3 2 3, repeat 9, 4 5 5, bleeding all, mobility class 3",
            "probing 3 3 3, 3 4 3, 4 3 3, 3 2 3, 2 2 3, repeat on 7, 3 2 3, 4 3 4, 3 3 4, 3 2 3, 3, distal suppuration and bleeding",
            "Mark number 18, furcation 2, 19, suppuration distal with proximal bleeding, 20, 4 3 3",
        ]

        # 학습에 사용하는 system prompt (data_loader.py의 _build_adaptive_cmd_prompt와 동일한 형식)
        system_prompt = """🦷 PERIODONTAL CHARTING ASSISTANT

TASK: Convert natural language into structured command sequences.

CRITICAL: Use ONLY commands from AVAILABLE COMMANDS MAP below.

TOOTH NUMBERING: UNS
[UNS] Q1(1-8), Q2(9-16), Q3(17-24), Q4(25-32)
Quadrant: Q1 → teeth 1–8, Q2 → 9–16, Q3 → 17–24, Q4 → 25–32

COMMON RULES:
- Single line output, semicolons (;) separate commands
- Always start with "number N"
- Three numbers = probing values (NOT tooth number)
- Never output meta-commands: expand "repeat", "others", "all" to explicit commands
- VALIDATION: Check that all commands in your output exist in AVAILABLE COMMANDS MAP above

"""

        try:
            # Chat template을 사용하여 system prompt와 user prompt 분리
            # tokenizer에 chat_template이 있는지 확인
            has_chat_template = hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None
            
            if has_chat_template:
                # Chat template 사용 (Qwen, Llama 등)
                messages_list = []
                for prompt in sample_prompts[:self.max_samples]:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Convert: {prompt}\n\nOutput (commands only):"}
                    ]
                    messages_list.append(messages)
                
                # 각 메시지를 chat template으로 변환
                prompts_to_process = []
                for messages in messages_list:
                    try:
                        # add_generation_prompt 파라미터 지원 여부 확인
                        prompt_text = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    except TypeError:
                        # add_generation_prompt가 지원되지 않는 경우
                        prompt_text = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False
                        )
                    prompts_to_process.append(prompt_text)
            else:
                # Chat template이 없는 경우 (fallback)
                # 일반적인 형식: system prompt + user prompt
                prompts_to_process = [
                    f"{system_prompt}Convert: {prompt}\n\nOutput (commands only):"
                    for prompt in sample_prompts[:self.max_samples]
                ]
            
            inputs = tokenizer(
                text=prompts_to_process, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Decode only the generated part
            input_ids_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, input_ids_len:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # 원본 프롬프트 (system prompt 없이)
            original_prompts = sample_prompts[:self.max_samples]
            
            for i, (full_prompt, original_prompt, generated_only) in enumerate(zip(prompts_to_process, original_prompts, generated_texts)):
                log_entry = {
                    "step": current_step,
                    "generation_step": self.eval_step_count,
                    "sample_index": i,
                    "original_prompt": original_prompt,  # system prompt 없는 원본
                    "full_prompt": full_prompt,  # system prompt 포함 전체
                    "generated": generated_only.strip(),
                }
                generation_logs.append(log_entry)
                print(f"📝 Sample {i+1}/{len(prompts_to_process)}: {generated_only.strip()[:60]}...")
                
        except Exception as e:
            logger.error(f"❌ Error during generation logging: {e}", exc_info=True)

        if generation_logs:
            log_file = os.path.join(self.output_dir, f"generation_log_step_{current_step}.json")
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_logs, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Generation logs saved to {log_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save generation logs: {e}")

        print(f"\n{'='*80}")
        print(f"✅ STEP {current_step} - Generation 테스트 완료")
        print(f"{'='*80}\n")

        if was_training:
            model.train()


class CustomGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer를 상속받은 커스텀 트레이너"""

    def __init__(
        self,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction|ComponentRewardWrapper] = None,
        generation_log_dir: str = "./generation_logs",
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 5,
        *args,
        **kwargs
    ):
        # model_init_kwargs is not expected by the parent class
        if "model_init_kwargs" in kwargs:
             kwargs.pop("model_init_kwargs",None)
        
        self.custom_reward_functions = reward_functions or []

        super().__init__(reward_funcs=self.custom_reward_functions, *args, **kwargs)
        
        self.add_callback(
            GenerationLoggingCallback(
                trainer=self,
                output_dir=generation_log_dir,
                max_samples=max_generation_samples,
                log_every_n_steps=generation_log_every_n_steps
            ))
        logger.info(f"✅ Generation logging callback added (every {generation_log_every_n_steps} steps)")

    def compute_rewards(
        self,
        completions,
        **kwargs
    ):
        if not self.custom_reward_functions:
            return super().compute_rewards(completions, **kwargs)

        # completions가 리스트가 아닌 경우 리스트로 변환
        if not isinstance(completions, list):
            completions = [completions]
        
        # completion 개수 확인
        num_completions = len(completions)
        num_generations = getattr(self, 'num_generations', 2)
        
        # 디버깅 정보
        logger.debug(
            f"🔍 compute_rewards called: {num_completions} completions, "
            f"num_generations={num_generations}, "
            f"num_reward_functions={len(self.custom_reward_functions)}"
        )
        
        # completion 개수가 num_generations의 배수가 아니면 경고 및 조정
        if num_completions % num_generations != 0:
            logger.warning(
                f"⚠️ Completion count ({num_completions}) is not a multiple of "
                f"num_generations ({num_generations}). Adjusting..."
            )
            # num_generations의 배수가 되도록 패딩 또는 잘라냄
            target_size = ((num_completions + num_generations - 1) // num_generations) * num_generations
            if num_completions < target_size:
                # 부족하면 마지막 completion을 복사하여 패딩
                padding_needed = target_size - num_completions
                last_completion = completions[-1] if completions else ""
                completions = completions + [last_completion] * padding_needed
                logger.info(f"📊 Padded completions from {num_completions} to {len(completions)}")
            else:
                # 많으면 잘라냄
                completions = completions[:target_size]
                logger.info(f"📊 Trimmed completions from {num_completions} to {len(completions)}")
            num_completions = len(completions)
        
        # 각 reward 함수에서 reward 계산
        all_rewards = []
        for reward_func in self.custom_reward_functions:
            try:
                rewards = reward_func(completions, **kwargs)
                # rewards가 리스트가 아니면 리스트로 변환
                if not isinstance(rewards, list):
                    rewards = [rewards]
                
                # 길이가 completion 개수와 일치하지 않으면 조정
                if len(rewards) != num_completions:
                    logger.warning(
                        f"⚠️ Reward function {reward_func} returned {len(rewards)} rewards "
                        f"but expected {num_completions}. Adjusting..."
                    )
                    if len(rewards) < num_completions:
                        # 부족하면 마지막 값으로 패딩
                        rewards = rewards + [rewards[-1] if rewards else 0.0] * (num_completions - len(rewards))
                    else:
                        # 많으면 잘라냄
                        rewards = rewards[:num_completions]
                
                all_rewards.append(rewards)
            except Exception as e:
                logger.error(f"❌ Error in reward function {reward_func}: {e}", exc_info=True)
                # 에러 발생 시 0으로 채운 리스트 반환
                all_rewards.append([0.0] * num_completions)

        if not all_rewards:
            logger.warning("⚠️ No rewards computed, using default")
            return super().compute_rewards(completions, **kwargs)
        
        # 모든 reward 함수의 결과를 평균
        # 각 completion에 대해 모든 reward 함수의 평균 계산
        final_rewards = []
        for i in range(num_completions):
            rewards_for_completion = [rewards[i] for rewards in all_rewards if i < len(rewards)]
            if rewards_for_completion:
                final_rewards.append(sum(rewards_for_completion) / len(rewards_for_completion))
            else:
                final_rewards.append(0.0)
        
        # 최종 reward 개수가 completion 개수와 일치하는지 확인
        if len(final_rewards) != num_completions:
            logger.error(
                f"❌ Final rewards count ({len(final_rewards)}) doesn't match "
                f"completion count ({num_completions})"
            )
            # 강제로 맞춤
            if len(final_rewards) < num_completions:
                final_rewards = final_rewards + [final_rewards[-1] if final_rewards else 0.0] * (num_completions - len(final_rewards))
            else:
                final_rewards = final_rewards[:num_completions]
        
        # 최종 검증: reward 개수가 num_generations의 배수인지 확인
        if len(final_rewards) % num_generations != 0:
            logger.error(
                f"❌ Final rewards count ({len(final_rewards)}) is not a multiple of "
                f"num_generations ({num_generations}). This will cause shape error!"
            )
            # 강제로 num_generations의 배수로 맞춤
            target_size = ((len(final_rewards) + num_generations - 1) // num_generations) * num_generations
            if len(final_rewards) < target_size:
                final_rewards = final_rewards + [final_rewards[-1] if final_rewards else 0.0] * (target_size - len(final_rewards))
            else:
                final_rewards = final_rewards[:target_size]
            logger.warning(f"⚠️ Adjusted rewards to {len(final_rewards)} to match num_generations")
        
        logger.debug(f"✅ compute_rewards returning {len(final_rewards)} rewards")
        return final_rewards

    def _prepare_inputs(self, inputs):
        if not self.model.training:
            # During evaluation, we need to duplicate the generation_batch by num_generations
            # to match the expected format for _generate_and_score_completions
            if isinstance(inputs, list) and len(inputs) > 0:
                # Check if this is a generation batch that needs duplication
                # The parent class expects num_generations duplicates of each prompt
                num_generations = getattr(self, 'num_generations', 2)
                original_size = len(inputs)
                
                # Check if inputs are already duplicated (size should be multiple of num_generations)
                if original_size % num_generations != 0:
                    # Duplicate each input num_generations times
                    duplicated_inputs = []
                    for item in inputs:
                        for _ in range(num_generations):
                            # Deep copy to avoid reference issues
                            duplicated_inputs.append(copy.deepcopy(item))
                    inputs = duplicated_inputs
                    logger.info(f"📊 Evaluation: Duplicated batch from {original_size} to {len(inputs)} items (num_generations={num_generations})")
            
            # During evaluation, check input type first
            # If inputs is a dict, move to device
            # If inputs is a list or other type, let parent class handle it
            if isinstance(inputs, dict):
                try:
                    device = next(self.model.parameters()).device
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                except StopIteration:  # No parameters
                    return inputs
            else:
                # For list or other types, use parent implementation
                # TRL GRPOTrainer may have special handling for these
                return super()._prepare_inputs(inputs)
        return super()._prepare_inputs(inputs)
        
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if model.training:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Evaluation 시: TRL GRPOTrainer는 generation과 reward 계산을 수행
        # parent의 prediction_step을 호출하면 _generate_and_score_completions가 호출됨
        # 이 과정에서 compute_rewards가 호출되므로, 우리가 수정한 compute_rewards가 사용됨
        
        try:
            # Prepare inputs (may return dict, list, or other types)
            inputs = self._prepare_inputs(inputs)
            
            # Check if inputs is a dict with labels
            if not isinstance(inputs, dict):
                # If inputs is not a dict (e.g., list), use parent implementation
                # TRL GRPOTrainer는 list 형태의 inputs를 처리할 수 있음
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            
            # If no labels, TRL GRPOTrainer는 generation을 수행해야 함
            # parent의 prediction_step을 호출하여 TRL의 로직 사용
            if 'labels' not in inputs:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            # Process dict inputs with labels (일반적인 loss 계산)
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.get("loss")
                logits = outputs.get("logits")
            
            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, logits, inputs.get('labels'))
            
        except RuntimeError as e:
            if "shape" in str(e) and "invalid for input of size" in str(e):
                # Reward 개수 불일치 에러 처리
                logger.error(
                    f"❌ Reward shape mismatch error in evaluation: {e}\n"
                    f"   This usually means the number of rewards doesn't match "
                    f"   num_generations * num_prompts. Check compute_rewards implementation."
                )
                # 에러를 다시 발생시켜서 상위에서 처리하도록 함
                raise
            else:
                # 다른 RuntimeError는 그대로 전파
                raise


class UnslothGRPOTrainWorkflow:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""

    def __init__(
        self,
        config: GRPOConfig,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction|ComponentRewardWrapper] = None,
        generation_log_dir: str = None,
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 10
    ):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        
        if reward_functions is None:
            logger.info("No reward functions provided, using CommandRewardFunction by default.")
            cmd_reward_func = CommandRewardFunction()
            self.reward_functions = cmd_reward_func.expand_to_individual_rewards()
        else:
            self.reward_functions = reward_functions
            
        self.generation_log_dir = generation_log_dir or os.path.join(config.output_dir, "generation_logs")
        self.max_generation_samples = max_generation_samples
        self.generation_log_every_n_steps = generation_log_every_n_steps
        self.trainer = None
        self._load_model()
        logger.info("✅ Unsloth GRPO Trainer initialized successfully")
    
    def _load_model(self):
        """Load model and tokenizer using Unsloth"""
        model_name = self.model_init_kwargs.get("model_name", "unsloth/Qwen3-0.6B-bnb-4bit")
        logger.info(f"🔄 Loading model: {model_name}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.max_prompt_length,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
            )

            # Decoder-only 모델에서 right padding은 generation 오류를 유발하므로 left padding으로 강제
            try:
                self.tokenizer.padding_side = "left"
                # pad_token이 없으면 eos_token으로 설정하여 패딩 시 토큰 손실 방지
                if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"⚠️ tokenizer padding_side 설정 중 경고: {e}")
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=[
                    "q_proj", #"k_proj", "v_proj", 
                    "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
                use_rslora=False,
                loftq_config={},
            )
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            raise
    
    def create_grpo_trainer(
        self,
        train_dataset,
        eval_dataset=None
    ):
        """Create TRL GRPOTrainer"""
        logger.info("🔄 Creating TRL GRPOTrainer")
        try:
            self.trainer = CustomGRPOTrainer(
                reward_functions=self.reward_functions,
                generation_log_dir=self.generation_log_dir,
                max_generation_samples=self.max_generation_samples,
                generation_log_every_n_steps=self.generation_log_every_n_steps,
                model=self.model,
                args=self.config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            logger.info("✅ TRL GRPOTrainer created successfully")
            return self.trainer
        except Exception as e:
            logger.error(f"❌ Failed to create GRPOTrainer: {e}", exc_info=True)
            raise
    
    def train(self, train_dataset, eval_dataset=None):
        logger.info("🚀 Starting GRPO training with TRL")
        try:
            trainer = self.create_grpo_trainer(train_dataset, eval_dataset)
            training_result = trainer.train()
            logger.info("✅ GRPO training completed successfully")
            return training_result
        except Exception as e:
            logger.error(f"❌ GRPO training failed: {e}", exc_info=True)
            raise

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir or self.config.output_dir
        logger.info(f"💾 Saving model to {output_dir}")
        try:
            if self.trainer and hasattr(self.trainer, 'model'):
                self.trainer.save_model(output_dir)
            else:
                self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"✅ Model and tokenizer saved successfully to {output_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}", exc_info=True)
            raise

    def evaluate(self, eval_dataset=None):
        logger.info("📊 Starting evaluation")
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call create_grpo_trainer() first.")
        try:
            eval_results = self.trainer.evaluate(eval_dataset)
            logger.info(f"📊 Evaluation completed: {eval_results}")
            return eval_results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
            raise

def create_grpo_trainer(
    config: GRPOConfig,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    reward_functions: Optional[List] = None,
    generation_log_dir: str = None,
    max_generation_samples: int = 5,
    generation_log_every_n_steps: int = 5
) -> UnslothGRPOTrainWorkflow:
    """Create GRPO trainer with given configuration, reward functions, and generation logging (항상 활성화)"""
    return UnslothGRPOTrainWorkflow(
        config=config,
        model_init_kwargs=model_init_kwargs,
        reward_functions=reward_functions,
        generation_log_dir=generation_log_dir,
        max_generation_samples=max_generation_samples,
        generation_log_every_n_steps=generation_log_every_n_steps)
