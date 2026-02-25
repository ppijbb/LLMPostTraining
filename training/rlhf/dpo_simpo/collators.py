from transformers import DataCollatorForLanguageModeling
import torch

class DPODataCallator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length,):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _tokenizing(self, text):
        return self.tokenizer(text,
                              truncation=True,
                              padding="max_length",
                              max_length=self.max_length,
                              return_tensors="pt")

    def __call__(self, batch):
        input_text = []
        labels = []
        for b in batch:
            input_text += [b["input_ids"]]
            labels += [b["labels"]]
        input_tokens = self._tokenizing(input_text)
        label_tokens = self._tokenizing(labels)

        return CollatorOutput(**{
            "prompt": input_tokens['promp'].to(self.device),
            "chosen": input_tokens['chosen'].to(self.device),
            "rejected": label_tokens['rejected'].to(self.device),
        })
        # raise Exception("STOP")


class CollatorOutput:
    def __init__(self, input_ids, attention_mask, labels):
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._labels = labels

    def __len__(self,):
        return len(self._input_ids)

    def __getitem__(self, key):
        match key:
            case "input_ids":
                return self._input_ids
            case "attention_mask":
                return self._attention_mask
            case "labels":
                return self._labels
            case _:
                raise KeyError(f"Key {key} not found")

    def __setitem__(self, key, value):
        match key:
            case "input_ids":
                self._input_ids = value
            case "attention_mask":
                self._attention_mask = value
            case "labels":
                self._labels = value
            case _:
                raise KeyError(f"Key {key} not found")

    def __iter__(self):
        return iter(self.__dict__.keys())

    def to_dict(self):
        return {
            "input_ids": self["input_ids"],
            "attention_mask": self["attention_mask"],
            "labels": self["labels"]
        }
    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, value):
        self._input_ids = value

    @property
    def attention_mask(self):
        return self._attention_mask

    @attention_mask.setter
    def attention_mask(self, value):
        self._attention_mask = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value
