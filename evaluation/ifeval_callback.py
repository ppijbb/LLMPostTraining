from typing import Dict, List, Optional
import json
import os
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# DeepSpeed 지원을 위한 유틸리티 함수
def get_inference_model_for_generate(model):
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

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets package not found. Please install it with:")
    print("pip install datasets")

try:
    from instruction_following_eval import evaluation_main
except ImportError:
    print("Warning: instruction_following_eval package not found. Please install it with:")
    print("pip install git+https://github.com/josejg/instruction_following_eval.git")


def write_pretty_json(filename, data):
    """Write data to JSON file with pretty formatting"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


class IFEvalCallback(TrainerCallback):
    """
    Callback for evaluating model's instruction following capabilities using Google's IFEval
    framework during training.
    
    This callback automatically loads the IFEval dataset from Hugging Face and uses it to
    evaluate the model's instruction following capabilities during training.
    """
    
    def __init__(
        self, 
        eval_dataset_name: str = "google/IFEval",
        max_samples: Optional[int] = None,
        generation_config: Optional[Dict] = None,
        template: str = "{prompt}",
        batch_size: int = 32,
        output_dir: str = "ifeval_results",
        eval_steps: Optional[int] = None,
    ):
        """
        Args:
            eval_dataset_name: Name of the dataset on Hugging Face. Default is "google/IFEval".
            max_samples: Maximum number of samples to use for evaluation. If None, uses all samples.
            generation_config: Configuration for text generation
            template: Template for formatting prompts, default is just using the raw prompt
            batch_size: Batch size for generating responses
            output_dir: Directory to save evaluation results
            eval_steps: Custom evaluation frequency in steps, if None uses trainer's eval_steps
        """
        self.eval_dataset_name = eval_dataset_name
        self.max_samples = max_samples
        self.generation_config = generation_config or {
            # "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": 0.1,
        }
        self.template = template
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.eval_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset 
        self.eval_dataset = None  # Will be loaded during initialization
        self._load_eval_dataset()
    
    def _load_eval_dataset(self):
        """Load the IFEval dataset from Hugging Face"""
        try:
            # Load the dataset
            dataset = load_dataset(self.eval_dataset_name)
            
            # Convert to list of dictionaries
            eval_data = []
            for item in dataset["train"]:
                eval_data.append({
                    "key": item["key"],
                    "instruction_id_list": item["instruction_id_list"],
                    "prompt": item["prompt"],
                    "kwargs": item["kwargs"]
                })
            
            # Limit number of samples if specified
            if self.max_samples is not None and self.max_samples > 0:
                eval_data = eval_data[:self.max_samples]
            
            self.eval_dataset = eval_data
            print(f"Successfully loaded {len(self.eval_dataset)} samples from {self.eval_dataset_name}")
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using empty dataset. Please check your internet connection or dataset name.")
            self.eval_dataset = []
    
    def prepare_prompts(self, prompts, tokenizer):
        """Prepare and tokenize prompts for generation"""
        # Save original padding side
        original_padding_side = tokenizer.padding_side
        
        # Set left padding for generation
        tokenizer.padding_side = "left"
        
        prompts_tok = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=False,
            pad_to_multiple_of=8,
            add_special_tokens=False
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        
        return prompts_tok

    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        model, 
        tokenizer,
        **kwargs
    ):
        """Run IFEval evaluation when trainer.evaluate() is called"""
        # Check if we should evaluate based on custom eval_steps
        if self.eval_steps is not None:
            if state.global_step % self.eval_steps != 0 and state.global_step != 0:
                return control

        # Increment evaluation counter
        self.eval_count += 1
        
        # Check if we have evaluation data
        if not self.eval_dataset or len(self.eval_dataset) == 0:
            print("No evaluation data available. Skipping IFEval evaluation.")
            return control
        
        # Prepare for evaluation
        model.eval()
        
        # DeepSpeed 모델 처리: 실제 모델 추출
        inference_model, gather_context = get_inference_model_for_generate(model)
        
        # Enable cache for faster generation
        use_cache_orig = getattr(inference_model.config, "use_cache", False)
        inference_model.config.use_cache = True
        
        print(f"\n===== Running IFEval evaluation ({self.eval_count}) =====")
        
        # Create a copy of the evaluation dataset to store responses
        responses = deepcopy(self.eval_dataset)
        
        # Create batches for efficient processing
        batches = [responses[i:i + self.batch_size] for i in range(0, len(responses), self.batch_size)]
        
        # Generate responses for each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating responses")):
            # Format prompts according to template
            prompts = [self.template.format(prompt=item["prompt"]) for item in batch]
            
            # Tokenize prompts
            prompts_tok = self.prepare_prompts(prompts, tokenizer)
            
            # Generate responses
            with torch.no_grad():
                # Get pad token ID with appropriate fallback options
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = getattr(inference_model.config, "pad_token_id", 0)
                
                # Add pad_token_id to generation config if not present
                gen_config = self.generation_config.copy()
                if "pad_token_id" not in gen_config:
                    gen_config["pad_token_id"] = pad_token_id
                
                # ZeRO Stage 3인 경우 GatheredParameters 컨텍스트 사용
                if gather_context is not None:
                    with gather_context:
                        outputs_tok = inference_model.generate(
                            **prompts_tok,
                            **gen_config
                        ).to("cpu")
                else:
                    outputs_tok = inference_model.generate(
                        **prompts_tok,
                        **gen_config
                    ).to("cpu")
            
            # Decode responses
            outputs = [
                tokenizer.decode(
                    outputs_tok[i][outputs_tok[i] != pad_token_id], 
                    spaces_between_special_tokens=False,
                    skip_special_tokens=False
                )
                for i in range(len(outputs_tok))
            ]
            
            # Store responses, trimming the prompt part
            for i, output_raw in enumerate(outputs):
                # Calculate the index in the overall dataset
                idx = batch_idx * self.batch_size + i
                
                # Extract just the response part by removing the prompt
                response = output_raw[len(prompts[i]):].strip()
                
                # Store in the responses
                responses[idx]["response"] = response
            
            # Clean up GPU memory
            del prompts_tok, outputs_tok

        # Process special tokens in responses if needed
        special_tokens = tokenizer.all_special_tokens
        for r in responses:
            # Remove any special tokens from the start/end of responses
            for token in special_tokens:
                if token in r["response"]:
                    r["response"] = r["response"].split(token)[0]

        # Run IFEval
        try:
            ife_results = evaluation_main.do_ife(response_dict=responses)
            
            # Log metrics to trainer
            if ife_results:
                for metric_name, value in ife_results.items():
                    if isinstance(value, (int, float)):
                        args.log({
                            f"ifeval/{metric_name}": value,
                            "step": state.global_step,
                            "epoch": state.epoch
                        })
            
            # Save detailed results to file
            checkpoint_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            
            # Use either checkpoint-specific path or global path
            if os.path.exists(checkpoint_path):
                results_file = os.path.join(checkpoint_path, f"ifeval_results.json")
            else:
                results_file = os.path.join(self.output_dir, f"ifeval_results_step_{state.global_step}.json")
            
            # Add responses to results
            ife_results["responses"] = responses
            ife_results["num_samples"] = len(responses)
            ife_results["global_step"] = state.global_step
            ife_results["epoch"] = state.epoch
            
            # Save to file
            write_pretty_json(results_file, ife_results)
            
            print(f"IFEval results: {ife_results}")
            print(f"Detailed results saved to {results_file}")
            
        except Exception as e:
            print(f"Error running IFEval: {e}")
        
        # Restore original use_cache setting
        inference_model.config.use_cache = use_cache_orig
        
        return control
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Optionally run evaluation when model is saved"""
        # If you want to run evaluation at save points, uncomment this
        # if args.save_strategy != "no" and args.save_steps > 0:
        #     if state.global_step % args.save_steps == 0:
        #         control.should_evaluate = True
        
        return control

# Usage example:
"""
from ifeval_callback import IFEvalCallback

# Add callback to trainer
trainer.add_callback(
    IFEvalCallback(
        max_samples=50,  # Optional: limit number of samples
        template="<s>[INST] {prompt} [/INST]",  # For Llama-style models
        batch_size=4,
        output_dir="ifeval_results"
    )
)
""" 