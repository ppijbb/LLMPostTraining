import torch
from datasets import load_dataset
from transformers import GemmaTokenizer, GemmaTokenizerFast

def no_system_template(tokenizer):
    no_system_templates = [GemmaTokenizer, GemmaTokenizerFast]
    for t in no_system_templates:
        yield isinstance(tokenizer, t)

def processing(sample,
               tokenizer):
    # c = sample["chosen"]
    # r = sample["rejected"]
    # chosen = c[-2:]
    # rejected = r[-2:]

    # # print("chosen", chosen) # this should be chosen
    # # print("rejected", rejected) # this should be rejected
    # assert chosen[0] == rejected[0], "prompt not matched"
    # prompt = chosen[0]


    # history = []
    # # print(len(c), len(r))
    # for i in range(0, len(c), 2):  # this should be added to prompt
    #     c_pair = c[i:i+2]
    #     r_pair = r[i:i+2]
    #     if c_pair[0] == r_pair[0] and c_pair[1] not in chosen:
    #         history += c_pair

    # history += [prompt]
    # return {
    #     "prompt": tokenizer.apply_chat_template([prompt], tokenize=True,),
    #     "chosen": tokenizer.apply_chat_template([chosen[1]], tokenize=True),
    #     "rejected": tokenizer.apply_chat_template([rejected[1]], tokenize=True),
    #  }
    
    if any(no_system_template(tokenizer)):
        for sam in sample["chosen"]:
            if sam["role"] == "system":
                # sam["role"] = "user"
                sam["content"] = "You are a good assistant. "+sam["content"]
            # elif "system" not in sam["role"]:
            #     sam["role"] = "user
                
        for sam in sample["prompt"]:
            if sam["role"] == "system":
                # sam["role"] = "user"
                sam["content"] = "You are a good assistant. "+sam["content"]
                
        for sam in sample["rejected"]:
            if sam["role"] == "system":
                # sam["role"] = "user"
                sam["content"] = "You are a good assistant. "+sam["content"]
        sample["chosen"] = sample["prompt"] + sample["chosen"]
        sample["rejected"] = sample["prompt"] + sample["rejected"]

    if len(sample["rejected"]) ==0:
        sample["rejected"] = [{"role":"assistant", "content":""}]
    
    return {
        "prompt": tokenizer.apply_chat_template(sample["prompt"], tokenize=False,),
        "chosen": tokenizer.apply_chat_template(sample["chosen"], tokenize=False),
        "rejected": tokenizer.apply_chat_template(sample["rejected"], tokenize=False),
     }

def get_dataset(
    dataset_name: str,
    tokenizer):
    raw_dataset = load_dataset(
        dataset_name,
        trust_remote_code=True,
        revision="main",  # tag name, or branch name, or commit hash
        )

    return {
        dataset: raw_dataset[dataset].map(
            processing,
            batched=False,
            remove_columns=[n for n in raw_dataset.column_names if n not in ["train", "test"]],
            fn_kwargs={"tokenizer": tokenizer,}) for dataset in ["train", "test"]
        }
