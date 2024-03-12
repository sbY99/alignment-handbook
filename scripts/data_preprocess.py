import pandas as pd
import itertools
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train = pd.read_csv('raw-data/train.csv')

# 같은 카테고리 안에서 질문 2개&그에 따른 답변 2개씩 조합
setlist = []

random.seed(0)
qs = ['질문_1', '질문_2']
categories = train['category'].unique()

# 1. 카테고리별로 질문 2개씩 조합
for category in tqdm(categories):
    category_idx = train[train['category'] == category]['id']
    # 2. row 2개씩 조합
    question_combinations = list(itertools.combinations(category_idx, 2))
    lens = len(question_combinations)
    
    for idx, (i,j) in enumerate(question_combinations):
        # 3. 질문 2개에 대한 답변 2개씩 조합
        for qi1, qi2 in itertools.product(qs, qs):
            q1 = train[train['id'] == i][qi1].values[0]
            q2 = train[train['id'] == j][qi2].values[0]
            a1 = train[train['id'] == i]['답변_1'].values[0]
            a2 = train[train['id'] == j]['답변_1'].values[0]
            setlist.append(pd.DataFrame({'질문': [q1+q2], '답변': [a1+a2], 'category': [category]}))
        
        
trainset = pd.concat(setlist).reset_index(drop=True)

train_ratio = 0.95  # 예시로 95%를 train 데이터로 사용
train_df, dev_df = train_test_split(trainset, test_size=1-train_ratio, random_state=42)

train_df.to_csv('data/train.csv')
dev_df.to_csv('data/eval.csv')