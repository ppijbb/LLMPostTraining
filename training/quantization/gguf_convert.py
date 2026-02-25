from unsloth import FastLanguageModel
import torch


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Gunulhona/Gemma-Ko-Merge-PEFT",
    trust_remote_code=True,
    use_cache=True, # Use cache for faster decoding
    max_seq_length=4096,
    dtype=torch.bfloat16, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=False, # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    # attn_implementation="flash_attention_2", 
)
model.save_pretrained_gguf(
    "/nas/conan/gguf", 
    tokenizer, 
    quantization_method="q4_k_m"
    )
