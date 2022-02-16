import os
import torch
import numpy as np
from logger import logger
import logger as logging
from collections import Counter
from data_loader import PCLDataset
from datasets import load_metric
from transformers import BartConfig, TrainingArguments, Trainer
from transformers import BartTokenizer, BartForSequenceClassification

logging.init_logger()

metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")

def bart_compute_metrics(eval_pred):
    (logits, _), labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels)

best_model = "models/checkpoint-11000"

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForSequenceClassification(BartConfig(num_labels=2)).from_pretrained(best_model)

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

with open(f"predictions/bart_augment_{f1}_{best_model}.txt", 'w') as f:
    for pi in preds:
        f.write(f'{pi}\n')