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

metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")


def bart_compute_metrics(eval_pred):
    (logits, _), labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels)


def train():
    # hyperparameters
    batch_size = 4
    num_epochs = 5
    weight_decay = 0.01
    lr = 1e-5
    
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForSequenceClassification(BartConfig(num_labels=2)).from_pretrained("facebook/bart-base")

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # model = RobertaForSequenceClassification.from_pretrained("roberta-base")

    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

    train_data = PCLDataset(tokenizer, dataset="train", is_augment=True)
    test_data = PCLDataset(tokenizer, dataset="test", is_augment=False)

    training_args = TrainingArguments(
        "models", 
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=lr,
        )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data, compute_metrics=bart_compute_metrics)

    os.environ["WANDB_DISABLED"] = "true"
    trainer.train()

    outputs = trainer.predict(test_data)
    label_ids = outputs.label_ids
    logits, _ = outputs.predictions
    # logits = outputs.predictions

    preds = np.argmax(logits, axis=-1)
    logger.info(f"Prediction distribution: {Counter(preds)}")
    
    f1 = metric_f1.compute(predictions=preds, references=label_ids)["f1"]
    precision = metric_precision.compute(predictions=preds, references=label_ids)["precision"]
    recall = metric_recall.compute(predictions=preds, references=label_ids)["recall"]
    
    logger.info(f"Evaluation result: F1 = {f1}, Precision = {precision}, Recall = {recall}")

    with open(f"predictions/bart_augment_{f1}.txt", 'w') as f:
        for pi in preds:
            f.write(f'{pi}\n')
            
            
if __name__ == "__main__":
    logging.init_logger()
    train()