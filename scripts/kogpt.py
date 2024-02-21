import argparse
import logging
import os

import numpy as np
import pandas as pd
from pytorch_lightning import loggers as pl_loggers
from datasets import Dataset
from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer

from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, DataCollatorWithPadding
from transformers import DataCollatorForLanguageModeling

def main():
    import random
    import numpy as np
    import torch

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='</s>', mask_token='<mask>')

    train_data = pd.read_csv('data/train_v4.csv')
    eval_data = pd.read_csv('data/eval_v4.csv')

    train_texts = []
    eval_texts = []

    for i in range(len(train_data)):
        row = train_data.iloc[i]
        question = row['질문']
        answer = row['답변']
        #train_texts.append(f"[INST]{question}[/INST]{answer}{tokenizer.eos_token}")
        train_texts.append(f"[{question}{tokenizer.eos_token}{answer}{tokenizer.eos_token}")

    for i in range(len(eval_data)):
        row = eval_data.iloc[i]
        question = row['질문']
        answer = row['답변']
        #eval_texts.append(f"[INST]{question}[/INST]{answer}{tokenizer.eos_token}")
        eval_texts.append(f"[{question}{tokenizer.eos_token}{answer}{tokenizer.eos_token}")

    train_dataset = Dataset.from_dict({'text':train_texts})
    eval_dataset = Dataset.from_dict({'text':eval_texts})

    def tokenizer_function(example): 
        return tokenizer(example['text'], truncation=True, max_length=1024)

    #raw_dataset의 map 함수를 사용해서 tokenizer_function을 모든 데이터에 적용(E)
    tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="model/kogpt2-v4",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=1200,
        logging_steps=1200,
        gradient_accumulation_steps=1,
        num_train_epochs=4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        save_steps=1200,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()
