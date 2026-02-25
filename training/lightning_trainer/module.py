import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding, DataCollatorForLanguageModeling
import lightning as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchmetrics.functional.text.rouge import rouge_score
from evaluate import load
from bitsandbytes.optim import AdamW, Lion

from model import peft_model, tokenizer
from dataset import dataset, max_source_length


class CallatorOutput:
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



class SumDataCallator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length,):
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return CallatorOutput(**{
            "input_ids": input_tokens['input_ids'],
            "attention_mask": input_tokens['attention_mask'],
            "labels": label_tokens['input_ids'],
        })
        # raise Exception("STOP")


class FTDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, data_collator, train_batch_size, eval_batch_size, training_args):
        super().__init__()
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.data_collator = SumDataCallator(tokenizer, max_length=max_source_length,)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.training_args = training_args

    def _get_dataloader(self, dataset, eval_mode: bool = False):
        return DataLoader(dataset=dataset,
                          batch_size=self.train_batch_size if eval_mode else self.eval_batch_size,
                          shuffle=not eval_mode,
                          num_workers=8,
                          collate_fn=self.data_collator)

    def train_dataloader(self):
        return self._get_dataloader(dataset=self.train_dataset)

    def val_dataloader(self):
        return self._get_dataloader(dataset=self.val_dataset, eval_mode=True)

    def test_dataloader(self):
        return self._get_dataloader(dataset=self.test_dataset, eval_mode=True)


class LLamaFTLightningModule(L.LightningModule):
    def __init__(self, data_collator, learning_rate: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = peft_model
        self.tokenizer = tokenizer
        # self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)
        self.learning_rate = learning_rate

    def _get_rouge_score(self, predictions, labels):
        generated_tokens = predictions.argmax(dim=-1)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return rouge_score(preds=decoded_preds, target=decoded_labels)

    def _log(self, log_name, value, batch_size):
        self.log(
            log_name,
            value if value.device == self.model.device else value.to(self.model.device),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True)

    def _batch_device_correction(self, batch):
        for k, v in batch.items():
            if v.device != self.model.device:
                batch[k] = v.to(self.model.device)
        return batch

    def forward(self, input_ids, attention_mask, labels):
        # print(input_ids.shape, input_ids.min(), input_ids.max())
        return self.model(**{
            "input_ids":input_ids, 
            "attention_mask":attention_mask, 
            "labels": labels
            })

    def training_step(self, batch, batch_idx):
        batch = self._batch_device_correction(batch)
        outputs = self(input_ids=batch.input_ids,
                       attention_mask=batch.attention_mask,
                       labels=batch.labels)
        rouge_score = self._get_rouge_score(outputs.logits, batch.labels)
        loss = outputs.loss
        self._log("train_loss", loss, self.trainer.datamodule.train_batch_size,)
        for k, v in rouge_score.items():
            self._log(f"train_{k}", v, self.trainer.datamodule.train_batch_size,)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._batch_device_correction(batch)
        outputs = self(input_ids=batch.input_ids,
                       attention_mask=batch.attention_mask,
                       labels=batch.labels)
        rouge_score = self._get_rouge_score(outputs.logits, batch.labels)
        val_loss = outputs.loss
        self._log("val_loss", val_loss, self.trainer.datamodule.eval_batch_size,)
        for k, v in rouge_score.items():
            self._log(f"val_{k}", v, self.trainer.datamodule.eval_batch_size,)


    def on_validation_epoch_end(self, *args,**kwargs):
        prompt_texts = [{
            "role": "system",
            "content": "You are helpful summary LLama"
        }]

        context = """
 아래 대화 내용을 요약해줘:
 영희: 안녕 철수야, 내일 오후에 바쁘니?
 철수: 바쁠것 같아.. 왜?
 영희: 내일 동물 보호소에 같이 갈래?
 철수: 뭐 하려고?
 영희: 아들한테 강아지 선물 하려고.
 철수: 좋은 생각이다. 그래 같이 가자.
 영희: 난 작은 강아지 한마리를 아들에게 사 줄까 해.
 철수: 그래 너무 크지 않은 녀석이 좋겠다.
 ---
 요약:
""" 
        chat_template = {
            "role": "user",
            "content": context
        }
        prompt_texts.append(chat_template)
     
        
        print(f'''
            ########### test summary ############
            {self.generate(prompt_texts)}
            #####################################
              ''')


    def generate(self, text):
        return self.tokenizer.decode(
            self.model.generate(
                tokenizer.apply_chat_template(
                    text, 
                    add_generateion_prompt=True, 
                    return_tensors="pt" ).to(self.model.device),
                max_new_tokens=100,
                )[0])

    def configure_optimizers(self):
        optimizer = Lion(params=self.model.parameters(),
                         lr=self.learning_rate,
                         weight_decay=0.01,
                         optim_bits=32,)
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=10,
                                                T_mult=2,
                                                eta_min=0.00001)
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,

            },
        }
