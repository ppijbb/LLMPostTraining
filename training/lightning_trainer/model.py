import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from peft import (inject_adapter_in_model, prepare_model_for_kbit_training, 
                  get_peft_model, replace_lora_weights_loftq)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


base_model_id = "meta-llama/Meta-Llama-3-8B" # @param ["Gunulhona/tb_pretrained_sts", "Gunulhona/tb_pretrained", "google/flan-t5-xxl", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2-7B-Instruct", "google/gemma-7b", "MLP-KTLim/llama-3-Korean-Bllossom-8B", "EleutherAI/polyglot-ko-12.8b", "vilm/vulture-40b", "arcee-ai/Arcee-Spark", "Qwen/Qwen2-1.5B-Instruct", "OuteAI/Lite-Mistral-150M", "google/gemma-2b-it"] {allow-input: true}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


peft_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    # quantization_config=bnb_config
    )

peft_model = prepare_model_for_kbit_training(peft_model)

# adapter configuration
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights="gaussian", #"gaussian", "pissa", "pissa_niter_{n}", "loftq", False
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False,
    use_dora=False,
    task_type="CAUSAL_LM",
)

# peft_model.add_adapter(lora_config, adapter_name="adapter_1")
inject_adapter_in_model(lora_config, peft_model, "adapter_1")
peft_model = get_peft_model(peft_model, lora_config)


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_special_tokens=True,
    trust_remote_code=True)
tokenizer.model_input_names=['input_ids', 'attention_mask']
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
