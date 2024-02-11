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
from trl import AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2

import torch
from trl import PPOTrainer
from trl import PPOConfig
from tqdm import tqdm

class PPODataset(torch.utils.data.Dataset):
    def __init__(self, path1, path2=None):
        df1 = pd.read_csv(path1)
        if path2:
            df2 = pd.read_csv(path2)
            df = pd.concat((df1, df2))
        else:
            df = df1
            
        self.dataset = df1
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "gogamza/kobart-base-v2",
            bos_token="<s>",
            eos_token="</s>",
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )
        self.max_seq_len = 256
        self.bos_token = '<s>'
        self.eos_token = '</s>'

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
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
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

    def __len__(self):
        return len(self.dataset)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

def get_rewards(embed_model, response_texts, labels):
    rewards = []
    for pred, label in zip(response_texts, labels):
        pred_embed = embed_model.encode(pred)
        label_embed = embed_model.encode(label)
    
        sample_score = cosine_similarity(label_embed, pred_embed)
        rewards.append(sample_score)
    return rewards
    
def main():
    model = AutoModelForCausalLMWithValueHead.from_pretrained('../model/kobart/checkpoint-2800/')
    tokenizer = PreTrainedTokenizerFast.from_pretrained( "gogamza/kobart-base-v2", bos_token="<s>", eos_token="</s>",unk_token='<unk>',pad_token='<pad>',mask_token='<mask>')   

    # Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
    embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    train_df = pd.read_csv('../data/train_v3.csv')
    eval_df = pd.read_csv('../data/eval_v3.csv')

    train_dataset=PPODataset('../data/train_v3.csv','../data/eval_v3.csv')

    config = PPOConfig(
        model_name="gogamza/kobart-base-v2",
        learning_rate=1e-4,
        batch_size=64,
        gradient_accumulation_steps=2,
        ppo_epochs=4,
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=train_dataset,
        tokenizer=tokenizer
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length":512
    }

    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader): 
            query_tensors = batch["input_ids"]
            query_tensors_for_input = []
            response_tensors = []
            for query_tensor in query_tensors:
                #### Get response from SFTModel
                response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
                response_tensors.append(response_tensor)
                query_tensors_for_input.append(query_tensor)
            
            #batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            labels =[tokenizer.decode(l.squeeze(), skip_special_tokens=True) for l in batch['decoder_input_ids']]
            
            #### Compute reward score
            rewards = get_rewards(embed_model, response_texts, labels)

            #pipe_outputs = reward_model(texts)
            #rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors_for_input, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

if __name__ == "__main__":
    main()
