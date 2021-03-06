import os
import numpy as np
from logger import logger
import logger as logging
from util import seed_everything, metric_recall, metric_precision, metric_f1
from data_loader import PCLDataset
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from transformers import BartTokenizer, BartForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from model import MyTrainer, RoBERTa_PCL


def bart_compute_metrics(eval_pred):
    (logits, _), labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels)


def train():
    ################ hyperparameters ###############
    exp_name = "bart_large_paraphrases"
    batch_size = 32
    num_epochs = 20
    weight_decay = 0.01
    warmup_ratio = 0 # 0.1
    lr = 1e-5
    max_seq_length = 128
    early_stop = 5
    eval_steps = 327
    ################################################
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=2)

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # model = RoBERTa_PCL.from_pretrained("roberta-base", num_labels=2)

    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

    train_data = PCLDataset(tokenizer, dataset="train", is_augment=True, max_length=max_seq_length)
    test_data = PCLDataset(tokenizer, dataset="test", is_augment=False, max_length=max_seq_length)

    training_args = TrainingArguments(
        exp_name, 
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps = eval_steps,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        metric_for_best_model = 'f1',
        eval_steps = eval_steps,
        )
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_data, 
        eval_dataset=test_data, 
        compute_metrics=bart_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop)],
        )

    os.environ["WANDB_DISABLED"] = "true"
    trainer.train()

    outputs = trainer.predict(test_data)
    label_ids = outputs.label_ids
    logits, _ = outputs.predictions
    # logits = outputs.predictions

    preds = np.argmax(logits, axis=-1)
    f1 = metric_f1.compute(predictions=preds, references=label_ids)["f1"]
    precision = metric_precision.compute(predictions=preds, references=label_ids)["precision"]
    recall = metric_recall.compute(predictions=preds, references=label_ids)["recall"]
    logger.info(f"F1: {f1}; Precision: {precision}; Recall: {recall}")

    # with open(f"predictions/bart_large_{f1:.2f}.txt", 'w') as f:
    #     for pi in preds:
    #         f.write(f'{pi}\n')
 
if __name__ == "__main__":
    logging.init_logger()
    seed_everything(42)
    train()
