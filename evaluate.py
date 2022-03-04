import os
import torch
import numpy as np
from logger import logger
import logger as logging
from collections import Counter
from data_loader import PCLDataset
from datasets import load_metric
from transformers import BartConfig, RobertaTokenizer, TrainingArguments, Trainer
from transformers import BartTokenizer, BartForSequenceClassification

from model import RoBERTa_PCL
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.init_logger()

metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")

best_model = "bart_large_paraphrases_2/checkpoint-1308"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForSequenceClassification.from_pretrained(best_model, num_labels=2)

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = RoBERTa_PCL.from_pretrained("roberta/checkpoint-500", num_labels=2)
model.to(device)

test_data = PCLDataset(tokenizer, dataset="test", is_augment=False)

trainer = Trainer(model=model)

outputs = trainer.predict(test_data)
label_ids = outputs.label_ids
logits, _ = outputs.predictions

preds = np.argmax(logits, axis=-1)
logger.info(f"Prediction distribution: {Counter(preds)}")

f1 = metric_f1.compute(predictions=preds, references=label_ids)["f1"]
precision = metric_precision.compute(predictions=preds, references=label_ids)["precision"]
recall = metric_recall.compute(predictions=preds, references=label_ids)["recall"]

logger.info(f"Evaluation result: F1 = {f1}, Precision = {precision}, Recall = {recall}")

# with open(f"predictions/{f1:.2f}_{precision:.2f}_{recall:.2f}.txt", 'w') as f:
#     for pi in preds:
#         f.write(f'{pi}\n')