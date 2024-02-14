# coding=utf-8
import os
from tqdm import tqdm
import argparse
from collections import defaultdict
from accelerate.utils import set_seed
from sentence_transformers import SentenceTransformer 

import numpy as np
import pandas as pd

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import warnings
warnings.filterwarnings('ignore')

# Argument parser

PROMPT_TEMPLATE = '[INST]{question}[/INST]'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='GAI-LLM/Yi-Ko-6B-mixed-v15')
    parser.add_argument('--adapter_path', type=str,
                        default='model/GAI-LLM-Yi-Ko-6B-mixed-v15-sft-qlora-v1')
    parser.add_argument('--max_length', type=int,
                        default=512)
    parser.add_argument('--output_path', type=str,
                        default='result/output.csv')
    parser.add_argument('--response_path', type=str,
                        default='result/response.txt')
    args = parser.parse_args()

    config = defaultdict()
    for arg, value in args._get_kwargs():
        config[arg] = value

    return args


def str_to_boolean(str):
    if str == 't' or str == 'T' or str == 'True':
        return True
    elif str == 'f' or str == 'F' or str == 'False':
        return False
    else:
        raise ValueError('String must be t or T for True and f or F for False')

def extract_text(input_string):
    index_t = input_string.find('[/INST]')
    if index_t != -1:  
        result = input_string[index_t + len('[/INST]'):]
    else: 
        raise Exception
    return result

def main():
    random_seed = 42
    set_seed(random_seed)

    # Hidden warning message
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    args = get_args()

    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    test_df = pd.read_csv('data/test_raw.csv')
    sub_df = pd.read_csv('data/sample_submission.csv')

    generated_sentence = []

    for i in tqdm(range(len(test_df))):
        q = test_df.iloc[i]['질문']
        prompt = PROMPT_TEMPLATE.format(question=q,
                                        sep_token=tokenizer.eos_token)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(input_ids=inputs, max_length=args.max_length, num_beams=5)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = extract_text(response)

        generated_sentence.append(response)

    encode_list=[]
    for i in range(len(generated_sentence)):
        embed=embed_model.encode(generated_sentence[i]) #주어진 모델로 인코딩
        encode_list.append(embed)
    
    for i in range(len(encode_list)):
        sub_df.loc[i, 'vec_0':'vec_511']=encode_list[i]

    sub_df.set_index('id',inplace=True)
    sub_df.to_csv(args.output_path)

    # for debugging
    file_name = args.response_path

    with open(file_name, 'w+') as file:
        file.write('\n'.join(generated_sentence))

if __name__ == '__main__':
    main()
