import os
import torch
import numpy as np
from util import labels2file
from data_loader import PCLDataset
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import BartTokenizer, BartForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
metric = load_metric("f1")


def bart_compute_metrics(eval_pred):
    (logits, _), labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForSequenceClassification.from_pretrained("facebook/bart-base")

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # model = RobertaForSequenceClassification.from_pretrained("roberta-base")

    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

    train_data = PCLDataset(tokenizer, dataset="train")
    test_data = PCLDataset(tokenizer, dataset="test")

    training_args = TrainingArguments(
        "models", 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        load_best_model_at_end=True,
        )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data, compute_metrics=bart_compute_metrics)

    os.environ["WANDB_DISABLED"] = "true"
    trainer.train()

    outputs = trainer.predict(test_data)
    label_ids = outputs.label_ids
    logits, _ = outputs.predictions
    # logits = outputs.predictions

    preds = np.argmax(logits, axis=-1)
    f1 = metric.compute(predictions=preds, references=label_ids)["f1"]

    with open(f"predictions/bart_baseline_{f1}.txt", 'w') as f:
        for pi in preds:
            f.write(f'{pi}\n')
            
            
if __name__ == "__main__":
    train()