import numpy as np
from datasets import load_dataset, concatenate_datasets

from evaluate import load
from model import tokenizer
from utils import preprocess_function


dataset_path = "Samsung/samsum" # @param ["Samsung/samsum", "emozilla/soda_synthetic_dialogue", "frcp/summary-alpaca-v01"] {allow-input: true}

raw_dataset = load_dataset(
  dataset_path,
  trust_remote_code=True,
  revision="main",  # tag name, or branch name, or commit hash
)

metric = load("rouge")
full_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])
tokenized_inputs = full_dataset.map(
    lambda x: tokenizer(x["dialogue"], truncation=True),
    batched=True,
    remove_columns=["dialogue", "summary"])

input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 100)) + int(np.percentile(input_lenghts, 10))
max_source_length = min(4096, max_source_length)

tokenized_targets = full_dataset.map(
    lambda x: tokenizer(x["summary"], truncation=True),
    batched=True,
    remove_columns=["dialogue", "summary"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 100)) + int(np.percentile(target_lenghts, 10))
max_target_length = min(4096, max_target_length)

dataset = raw_dataset.map(preprocess_function,
                          batched=True,
                          remove_columns=["dialogue", "summary", "id"],
                          fn_kwargs={
                              "max_source_length": max_source_length,
                              "max_target_length": max_target_length
                             },)
# dataset = raw_dataset
# if any([d for d in dataset.values() if "token_type_ids" in d.features]):
#     dataset = dataset.map(lambda x: x,
#                           batched=True,
#                           remove_columns=["token_type_ids"], )
