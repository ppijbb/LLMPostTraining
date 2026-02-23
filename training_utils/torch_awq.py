import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

USER_ID = "Gunulhona"
model_id = f"{USER_ID}/Gemma-3-27B-v2"
model_to_quantize = f"{USER_ID}/Gemma-3-27B-v2"
from torchao.quantization import Int4WeightOnlyConfig, quantize_, ModuleFqnToConfig
from torchao.prototype.awq import (
    AWQConfig,
)
from torchao._models._eval import TransformerEvalWrapper
model = AutoModelForCausalLM.from_pretrained(
    model_to_quantize,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
def get_quant_config(linear_config):
    return ModuleFqnToConfig({
        r"re:language_model\.model\.layers\..+\.mlp\..+_proj": linear_config,
        r"re:language_model\.model\.layers\..+\.self_attn\..+_proj": linear_config,
        r"re:model\.language_model\.layers\..+\.mlp\..+_proj": linear_config,
        r"re:model\.language_model\.layers\..+\.self_attn\..+_proj": linear_config,
    })
# AWQ only works for H100 INT4 so far
base_config = Int4WeightOnlyConfig(group_size=128)
linear_config = AWQConfig(base_config, step="prepare")
quant_config = get_quant_config(linear_config)
quantize_(
    model,
    quant_config,
)
tasks = ["mmlu_philosophy"]
calibration_limit=30
max_seq_length=2048
TransformerEvalWrapper(
    model=model,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
).run_eval(
    tasks=tasks,
    limit=calibration_limit,
)
linear_config = AWQConfig(base_config, step="convert")
quant_config = get_quant_config(linear_config)
quantize_(model, quant_config)
quantized_model = model
linear_config = AWQConfig(base_config, step="prepare_for_loading")
quant_config = get_quant_config(linear_config)
quantized_model.config.quantization_config = TorchAoConfig(quant_config)

# Push to hub
MODEL_NAME = model_id.split("/")[-1]
save_to = f"{USER_ID}/{MODEL_NAME}-AWQ-INT4"
quantized_model.push_to_hub(save_to, safe_serialization=False)
tokenizer.push_to_hub(save_to)

# Manual Testing
quantized_model = AutoModelForCausalLM.from_pretrained(
    save_to,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {
        "role": "system",
        "content": "",
    },
    {"role": "user", "content": prompt},
]
templated_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("Prompt:", prompt)
print("Templated prompt:", templated_prompt)
inputs = tokenizer(
    templated_prompt,
    return_tensors="pt",
).to("cuda")
generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
output_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Response:", output_text[0][len(prompt):])
