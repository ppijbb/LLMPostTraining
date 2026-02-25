from typing import List, Dict, Any
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import torch

# 커스텀 콜백 정의
class TrainerDebugCallback(TrainerCallback):
    # def __init__(self, model, tokenizer):
    #     super().__init__()
    #     self.model = model
    #     self.tokenizer = tokenizer
    
    def _eval_generation(self, model:Any, tokenizer:Any, test_input: List[Dict]):
        with torch.inference_mode():
            print(
                tokenizer.decode(
                    model.generate(
                        input_ids=tokenizer.apply_chat_template(
                            test_input, 
                            tokenize=True, 
                            add_generation_prompt=True,
                            return_tensors="pt").to(model.device),
                        do_sample=False, 
                        temperature=1.0,
                        max_length=128
                        )[0],
                    skip_special_tokens=True
                    )
                )

    def on_step_begin(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Starting step {state.global_step}")
        pass
    
    def on_step_end(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Finished step {state.global_step}")
        pass
    
    def on_epoch_begin(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        print(f"Starting epoch {state.epoch}")
        # pass
        
    def on_epoch_end(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        print(f"Predictions in epoch {state.epoch}")
        test_input = [{"role": "user", "content": "What is the capital of France?"}]
        self._eval_generation(kwargs["model"], kwargs["processing_class"], test_input)
        # pass
        
    def on_log(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
