import torch
from collections import defaultdict
from outlines.models import TransformersMultiModal
from typing import List, Any, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers.processing_utils import ProcessorMixin
from transformers.generation.logits_process import LogitsProcessorList
from deepeval.models.base_model import DeepEvalBaseLLM
from outlines.inputs import Audio, Chat, Image, Video
from outlines.processors import OutlinesLogitsProcessor

# DeepSpeed 지원을 위한 유틸리티 함수
def get_inference_model(model):
    """
    DeepSpeed로 감싸진 모델에서 실제 모델을 추출합니다.
    ZeRO Stage 3의 경우 GatheredParameters 컨텍스트를 반환합니다.
    
    Returns:
        tuple: (actual_model, context_manager) 또는 (actual_model, None)
    """
    # DeepSpeed로 감싸진 모델인지 확인
    if hasattr(model, 'module'):
        # DeepSpeed engine이 있는지 확인
        if hasattr(model, 'engine'):
            engine = model.engine
            # ZeRO stage 확인
            try:
                zero_stage = engine.zero_optimization_stage()
            except:
                zero_stage = 0
            
            # ZeRO Stage 3인 경우 GatheredParameters 필요
            if zero_stage == 3:
                try:
                    import deepspeed
                    return model.module, deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=None)
                except ImportError:
                    # deepspeed가 없으면 그냥 module 사용
                    return model.module, None
            else:
                # ZeRO 0, 1, 2는 그냥 module 사용
                return model.module, None
        else:
            # DDP 등 다른 래핑
            return model.module, None
    else:
        # 래핑되지 않은 모델
        return model, None
# from ..models import G3MoEForCausalLM, G3MoEConfig, Gemma3Config
# If this import fails, try using an absolute import based on your project structure:
try:
    from models import G3MoEForCausalLM, G3MoEConfig, G3MoEConfig
    from transformers import Gemma3Config
except ImportError:
    # Fallback: add parent directory to sys.path and retry
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models import G3MoEForCausalLM, G3MoEConfig, G3MoEConfig
    from transformers import Gemma3Config

# Setting high precision for matmul
torch.set_float32_matmul_precision('high')


class CustomOutlinesTransformers(TransformersMultiModal):
    @torch.no_grad()
    def _prepare_model_inputs(
        self,
        model_input,
        is_batch: bool = False,
    ) -> Tuple[Union[str, List[str]], dict]:
        """Turn the user input into arguments to pass to the model"""
        if is_batch:
            prompts = [
                self.type_adapter.format_input(item) for item in model_input
            ]
        else:
            prompts = self.type_adapter.format_input(model_input)

        # The expected format is a single dict
        if is_batch:
            merged_prompts = defaultdict(list)
            for d in prompts:
                for key, value in d.items():
                    if key == "text":
                        merged_prompts[key].append(value)
                    else:
                        merged_prompts[key].extend(value)
        else:
            merged_prompts = prompts # type: ignore

        inputs = self.processor(
            **merged_prompts, padding=True, return_tensors="pt"
        ).to(self.model.device)

        return merged_prompts["text"], inputs
    
    @torch.no_grad()
    def _generate_output_seq(self, prompts, inputs, **inference_kwargs):
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        if "token_type_ids" in inference_kwargs:
            del inference_kwargs["token_type_ids"]
        input_ids = inputs["input_ids"]
        
        # DeepSpeed 모델 처리: 실제 모델 추출
        inference_model, gather_context = get_inference_model(self.model)
        
        # ZeRO Stage 3인 경우 GatheredParameters 컨텍스트 사용
        if gather_context is not None:
            with gather_context:
                output_ids = inference_model.generate(
                    **inputs,
                    **inference_kwargs,
                )
        else:
            output_ids = inference_model.generate(
                **inputs,
                **inference_kwargs,
            )

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if inference_model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        return generated_ids


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str | PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        *args,
        **kwargs
    ):
        super().__init__(model_name=model, *args, **kwargs)
        if isinstance(model, str):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
            config = AutoConfig.from_pretrained(model).to_dict()
            if "text_config" in config or "vision_config" in config or "audio_config" in config:
                self.tokenizer = AutoProcessor.from_pretrained(model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model)

        elif tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device if hasattr(self.model, 'device') else "cuda"
        else:
            raise ValueError("Either model or tokenizer must be provided")
        # outlines.from_transformers(self.model, self.tokenizer)
        self.structured_model = CustomOutlinesTransformers(self.model, self.tokenizer)

    def load_model(self):
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @torch.no_grad()
    def generate(self, prompt: str, schema: Optional[Any] = None, **kwargs) -> str:
        """
        Generate text from prompt, handling DeepEval's prompt and schema parameters
        CRITICAL: Ensure model is in eval mode to avoid gradient computation
        """
        # CRITICAL: Ensure model is in eval mode (no gradients)
        original_training = None
        if hasattr(self.model, 'training'):
            original_training = self.model.training
            self.model.eval()
        
        # Remove DeepEval-specific parameters that the model doesn't expect
        model_kwargs = kwargs.copy()
        _prompt = model_kwargs.get("prompt", None)
        if schema is None:
            schema = model_kwargs.get("schema", None)
        try:
            response = self.structured_model(
                model_input={"text": prompt}, 
                output_type=schema,
                generation_config=GenerationConfig(
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id if isinstance(self.tokenizer, PreTrainedTokenizer) else self.tokenizer.tokenizer.eos_token_id
                ))
            return schema.model_validate_json(response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Generation error: {e}")
            return f"Error during generation: {str(e)}"
        finally:
            # Restore original training state if changed
            if original_training is not None and hasattr(self.model, 'train'):
                self.model.train(original_training)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str], schemas: Optional[Any] = None, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts
        """
        results = []
        for prompt, schema in zip(prompts, schemas):
            try:
                result = self.generate(prompt, schema, **kwargs)
                results.append([result])
            except Exception as e:
                print(f"Error generating for prompt: {e}")
                results.append(f"Error: {str(e)}")
        return results

    def get_model_name(self):
        if isinstance(self.model_name, str):
            return self.model_name
        else:
            return getattr(self.model, 'name_or_path', 'unknown_model')


def run_deepeval()->None:
    try:
        from deepeval.benchmarks import HellaSwag
        from deepeval.benchmarks import MMLU
        from deepeval.benchmarks.mmlu.task import MMLUTask
        from deepeval.benchmarks import BigBenchHard
        from deepeval.benchmarks import HumanEval
        from deepeval.benchmarks.tasks import HumanEvalTask
        from deepeval.benchmarks import SQuAD
        from deepeval.benchmarks.tasks import SQuADTask
        from deepeval.benchmarks import GSM8K
        from deepeval.benchmarks import MathQA
        from deepeval.benchmarks.tasks import MathQATask

        benchmarks = [
            MMLU(
                tasks=[task for task in MMLUTask],
                n_shots=5,
                n_problems_per_task=3
            ),
            # HellaSwag(n_shots=3),
            # BigBenchHard(enable_cot=True),
            # HumanEval(
            #     tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
            #     n=100
            # ),
            # SQuAD(
            #     tasks=[SQuADTask.PHARMACY, SQuADTask.NORMANS],
            #     n_shots=3
            # ),
            # GSM8K(
            #     n_problems=10,
            #     n_shots=3,
            #     enable_cot=True
            # ),
            #  MathQA(
            #     tasks=[MathQATask.PROBABILITY, MathQATask.GEOMETRY],
            #     n_shots=3
            # )
        ]
            
        model_name = "Gunulhona/Gemma-3-4B"
        model_architecture = G3MoEForCausalLM ## G3MoEForCausalLM
        base_config = Gemma3Config.from_pretrained(model_name)
        base_config = base_config.to_dict()
        moe_config = {
                "n_shared_experts": 1,
                "n_routed_experts": 5, # 256, 15, 6
                "n_group": 4,
                "topk_group": 8,
                # "num_key_value_heads": base_config['text_config']['num_attention_heads'],
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 18,
                "router_aux_loss_coef": 0.001,
                "router_jitter_noise": 0.01,
                "input_jitter_noise": 0.01,
                "model_type": "g3moe_text",
                "no_rope_layer_interval": 0,
                "rope_scaling":{
                    "rope_type": "yarn",
                    "factor": 8.0
                },
                # "intermediate_size": base_config['text_config']['hidden_size'],
                "use_bfloat16": True,
            }
        base_config['text_config'].update(moe_config)
        base_config.update(base_config['text_config'])
        model_config = G3MoEConfig(**base_config)
        model_config.model_type = "g3moe"
        # BitsAndBytesConfig int-4 config
        test_model = model_architecture.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_quant_storage=torch.bfloat16)
            ).to("cuda")
        tokenizer = AutoProcessor.from_pretrained(model_name)
        eval_model = LocalModel(model=test_model, tokenizer=tokenizer)

        for benchmark in benchmarks:
            results = benchmark.evaluate(model=eval_model, batch_size=8)
            print(f"Overall Score: {benchmark.__class__.__name__} - {results}")

    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        print("--- Done ---")
    
    
if __name__ == "__main__":
    run_deepeval()
