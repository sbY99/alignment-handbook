# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from sentence_transformers import SentenceTransformer

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import pandas as pd
import json
from datasets import Dataset
from trl import PPOConfig

import warnings
warnings.filterwarnings('ignore')

#tqdm.pandas()

trl_model_class = AutoModelForCausalLMWithValueHead
embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def extract_text(input_string):
    index_t = input_string.find('[/INST]')
    if index_t != -1:  
        result = input_string[index_t + len('[/INST]'):]
    else: 
        raise Exception
    return result


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


def compute_rewards(preds, labels):
    rewards = []

    for pred, label in zip(preds, labels):
        pred_embed = embed_model.encode(pred)

        label_embed_list = []
        for l in label:
            label_embed_list.append(embed_model.encode(l))
        
        cos_list = [cosine_similarity(pred_embed, l_e) for l_e in label_embed_list]
        mean_cos = sum(cos_list)/len(cos_list)

        rewards.append(torch.tensor(mean_cos))
    
    return rewards

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(data_path, model_name):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    
    questions = []
    label_dict = {}

    for item in data['data']:
        questions.append(
            f"[INST]{item['질문']}[/INST]"
        )
        label_dict[item['질문']] = item['답변']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({'text':questions})

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = dataset.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds, label_dict

data_path = "data/train_ppo_v2.json"
model_name = "yanolja/KoSOLAR-10.7B-v0.2"
adapter_path = 'model/yanolja-KoSOLAR-10.7B-v0.2-sft-qlora-v5'

# We retrieve the dataloader by calling the `build_dataset` function.
dataset, label_dict = build_dataset(data_path, model_name)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(42)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
ref_model = None
# Copy the model to each device
device_map = {"": Accelerator().local_process_index}

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    seed=42,
    batch_size=2
)

model = trl_model_class.from_pretrained(
    adapter_path,
    trust_remote_code=True,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
}

device = ppo_trainer.accelerator.device

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu" 

for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query = batch['query']
    query_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    #batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    labels = []
    for text in query:
        labels.append(
            label_dict[text[len('[INST]'):-len('[/INST]')]]
        )

    rewards = compute_rewards(batch["response"], labels)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)


ppo_trainer.save_model(f"{adapter_path}-ppo-v2")