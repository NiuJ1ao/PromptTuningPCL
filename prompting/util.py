import os
import sys
from datasets import load_metric
sys.path.append('..')
import random
import numpy as np
import pandas as pd
import torch
from dont_patronize_me import DontPatronizeMe
from openprompt.data_utils import InputExample


metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")

class PromptUtil:

    template = '{"placeholder":"text_a"} It was {"mask"} .'
    verbalizer = {
        # 0: ["respectful", "friendly", "great"],
        # 1: ["patronizing", "condescending", "terrible"],
        0: ["respectful"],
        1: ["patronizing"],
        # 0: ["great"],
        # 1: ["terrible"],
    }
    
    def data2examples(df):
        df = df.reset_index()
        examples = []
        for _, row in df.iterrows():
            examples.append(InputExample(text_a=row["text"], label=row["label"], meta={"keyword": row["keyword"], "country": row["country"]}))
        return examples
    
    def test2examples(df):
        df = df.reset_index()
        examples = []
        for _, row in df.iterrows():
            examples.append(InputExample(text_a=row["text"], meta={"keyword": row["keyword"], "country": row["country"]}))
        return examples

class IncontextUtil:
    
    neg_label_words = ["respectful"]
    pos_label_words = ["patronizing"]
    
    template = '{"placeholder":"text_a"} It was {"mask"} </s> {"meta": "pos"} </s> {"meta": "neg"}'
    verbalizer = {
        0: neg_label_words,
        1: pos_label_words,
    }
    
    def data2examples(df, pos_df, neg_df):
        df = df.reset_index()
        examples = []
        for _, row in df.iterrows():
            # sample one example from both pos and neg respectively
            pos = pos_df.sample(1)
            neg = neg_df.sample(1)
            pos_label = random.choice(IncontextUtil.pos_label_words)
            neg_label = random.choice(IncontextUtil.neg_label_words)
            examples.append(InputExample(text_a=row["text"], label=row["label"], meta={"keyword": row["keyword"], "country": row["country"], "pos": f"{pos.text} It was {pos_label} .", "neg": f"{neg.text} It was {neg_label} ."}))
        return examples
    
    def test2examples(df, pos_df, neg_df):
        df = df.reset_index()
        examples = []
        for _, row in df.iterrows():
            # sample one example from both pos and neg respectively
            pos = pos_df.sample(1)
            neg = neg_df.sample(1)
            pos_label = random.choice(IncontextUtil.pos_label_words)
            neg_label = random.choice(IncontextUtil.neg_label_words)
            examples.append(InputExample(text_a=row["text"], meta={"keyword": row["keyword"], "country": row["country"], "pos": f"{pos.text} It was {pos_label} .", "neg": f"{neg.text} It was {neg_label} ."}))
        return examples

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def merge_augments(train_df):
    paraphrases = DontPatronizeMe('../data', '')
    paraphrases.load_task1(file_name="augment_positive_paraphrase.tsv")
    paraphrases = paraphrases.train_task1_df
    train_df = pd.concat([train_df, paraphrases])
    # eda = DontPatronizeMe('../data', '')
    # eda.load_task1(file_name="augment_positive_EDA.tsv")
    # eda = eda.train_task1_df
    # train_df = pd.concat([train_df, eda])
    return train_df

def downsampling(df):
    pcldf = df[df.label==1]
    npos = len(pcldf)
    return pd.concat([pcldf, df[df.label==0][:npos*4]])


def load_references(dataset):
    references = []
    # with open("../data/sentences_and_labels_test.txt", 'r') as f:
    #     for line in f:
    #         segs = line.strip().split("\t")
    #         if len(segs) == 1:
    #             references.append(int(segs[0]))
    #         else:
    #             references.append(int(segs[1]))
    for data in dataset:
        references.append(data.label)
    return references
