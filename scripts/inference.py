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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='yanolja/EEVE-Korean-10.8B-v1.0')
    parser.add_argument('--adapter_path', type=str,
                        default='model/final-model')
    parser.add_argument('--test_data_path', type=str,
                        default='raw-data/test.csv')
    parser.add_argument('--submission_data_path', type=str,
                        default='raw-data/sample_submission.csv')
    parser.add_argument('--max_new_tokens', type=int,
                        default=512)
    parser.add_argument('--output_path', type=str,
                        default='result/output.csv')
    parser.add_argument('--response_path', type=str,
                        default='result/output-text.txt')
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
    index_t = input_string.rfind('assistant')
    if index_t != -1:  
        result = input_string[index_t + len('assistant'):]
    else: 
        raise Exception
    return result.strip()

def main():
    random_seed = 42
    set_seed(random_seed)

    if not os.path.exists('result'):
        os.makedirs('result')

    # Hidden warning message
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    args = get_args()

    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map = "auto")
    model = PeftModel.from_pretrained(model, args.adapter_path, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    embed_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    test_df = pd.read_csv(args.test_data_path)
    sub_df = pd.read_csv(args.submission_data_path)

    generated_sentence = []

    for i in tqdm(range(len(test_df))):
        q = test_df.iloc[i]['질문']
       
        inputs = tokenizer.apply_chat_template(
            [
                {'role':'user','content':q}
            ]
            , add_generation_prompt=True,
            return_tensors='pt'
        ).to(device)

        outputs = model.generate(input_ids=inputs, 
                                 num_beams=5,
                                 eos_token_id=tokenizer.eos_token_id, 
                                 max_new_tokens=args.max_new_tokens)
        
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

    dataset = {'data':[]}
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        question = row['질문']
        response = generated_sentence[i]
        dataset['data'].append({
            'question':question,
            'response':response
        })

    import json
    with open(file_name, 'w') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
