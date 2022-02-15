from data_loader import Dataset
from datasets import load_metric, list_metrics
from transformers import BartTokenizer, BartForSequenceClassification, TrainingArguments, Trainer

import torch
import numpy as np

'''
To define custom loss function
'''
# class BartTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def compute_loss(self, model, inputs):
#         custom_loss = ...
#         return custom_loss

# print(list_metrics())

f1 = load_metric("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1.compute(predictions=predictions, references=labels)


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large")

train_data = Dataset(train=True)
test_data = Dataset(train=False)


training_args = TrainingArguments("BART PCL", evaluation_strategy="epoch")
trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data, compute_metrics=compute_metrics)

trainer.train()