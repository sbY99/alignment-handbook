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
from transformers import set_seed

import pandas as pd
import json
from datasets import Dataset
from trl import PPOConfig

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings('ignore')

#tqdm.pandas()

data_path = "data/train_ppo_raw_v1.json"
model_name = "yanolja/KoSOLAR-10.7B-v0.2"
adapter_path = 'model/yanolja-KoSOLAR-10.7B-v0.2-sft-qlora-v5'

device = "cuda:1"

model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, adapter_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1').to(device)

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


def make_dpo_data(preds, labels):
    chosen = ''
    rejected = ''
    max_score = 0
    min_score = 9999
    
    pred_embed_list = []
    for pred in preds:
        embed = embed_model.encode(pred)
        pred_embed_list.append(embed)

    label_embed_list = []
    for label in labels:
        embed = embed_model.encode(label)
        label_embed_list.append(embed)

    for idx, p_e in enumerate(pred_embed_list):
        cos_list = [cosine_similarity(p_e, l_e) for l_e in label_embed_list]
        mean_cos = sum(cos_list)/len(cos_list)

        if mean_cos > max_score:
            chosen = preds[idx]
            max_score = mean_cos
        if mean_cos < min_score:
            rejected = preds[idx]
            min_score = mean_cos
        
    return chosen, rejected

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

# We retrieve the dataloader by calling the `build_dataset` function.
dataset, label_dict = build_dataset(data_path, model_name)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(42)
tqdm.pandas()

generation_kwargs = {
    "num_beams": 6,
    "num_beam_groups":3,
    "num_return_sequences":6,
    "diversity_penalty":0.5,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
}

dpo_dataset = {'data':[]}

for idx, data in enumerate(tqdm(dataset)):
    query = data['query']
    query_tensors = data["input_ids"].reshape(1,-1).to(device)

    response_tensors = model.generate(
        input_ids=query_tensors,
        **generation_kwargs
    )
    
    response = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    #batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    query_text = query[len('[INST]'):-len('[/INST]')]

    preds = [extract_text(i) for i in response]
    labels = label_dict[query_text]
        
    chosen, rejected = make_dpo_data(preds=preds, labels=labels)
    item = {
        'prompt':query_text,
        'chosen':chosen,
        'rejected':rejected
    }
    dpo_dataset['data'].append(item)


with open("data/dpo/yanolja-KoSOLAR-10.7B-v0.2-raw-v1.json", "w") as json_file:
    json.dump(dpo_dataset, json_file, ensure_ascii=False, indent=4)
