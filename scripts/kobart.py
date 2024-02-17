import argparse
import logging
import os

import numpy as np
import pandas as pd
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

class ChatDataset(Dataset):
    def __init__(self, filepath, max_seq_len=128) -> None:
        self.filepath = filepath
        self.dataset = pd.read_csv(self.filepath)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "gogamza/kobart-base-v2",
            bos_token="<s>",
            eos_token="</s>",
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )
    def __len__(self):
        return len(self.dataset)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        record = self.dataset.iloc[index]
        q, a = record['질문'], record['답변']
        q_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(a) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            q_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            a_tokens, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_seq_len + 1)])
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}
    
def main():
    import random
    import numpy as np
    import torch

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained( "gogamza/kobart-base-v2", bos_token="<s>", eos_token="</s>",unk_token='<unk>',pad_token='<pad>',mask_token='<mask>')

    train_data= ChatDataset('data/train_v4.csv',512)
    val_data= ChatDataset('data/eval_v4.csv',512)

    training_args = TrainingArguments(
        output_dir="model/kobart-v4",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=800,
        logging_steps=800,
        gradient_accumulation_steps=1,
        num_train_epochs=4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        save_steps=800,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()

if __name__ == "__main__":
    main()
