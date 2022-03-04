import os
import sys
sys.path.append('..')
from logger import logger
import logger as logging
from dont_patronize_me import DontPatronizeMe
import torch
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from torch.nn import CrossEntropyLoss
from prompting.util import merge_augments, downsampling, seed_everything, load_references
from prompting.util import metric_f1, metric_precision, metric_recall
from prompting.util import PromptUtil

def train(seed):
    ################ hyperparameters ###############
    train_batch_size = 32
    val_batch_size = 32
    num_epochs = 20
    weight_decay = 0.01
    warmup_ratio = 0 # 0.1
    lr = 1e-5
    max_seq_length = 128
    early_stop_steps = 5
    model_name = "bart"
    model_path = "facebook/bart-large"
    exp_name = "prompt-tuning-paraphrases"
    ################################################

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    classes = [0, 1]
    
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    
    promptTemplate = ManualTemplate(
        text = PromptUtil.template,
        tokenizer = tokenizer,
    )
        
    train_dpm = DontPatronizeMe('../data', '')
    train_dpm.load_task1("train.tsv")
    train_df = train_dpm.train_task1_df
    # train_df = downsampling(train_df)
    train_df = merge_augments(train_df)
    
    train_dataset = PromptUtil.data2examples(train_df)
    logger.info(f"Positive data size: {len(train_df[train_df.label==1])}; Negative data size: {len(train_df[train_df.label==0])}")
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'].to_numpy())
    # logger.info(f"Class weights: {class_weights}")
    
    val_dpm = DontPatronizeMe('../data', '')
    val_dpm.load_task1("test.tsv")
    val_df = val_dpm.train_task1_df
    val_dataset = PromptUtil.data2examples(val_df)
    
    references = load_references(val_dataset)
    
    train_dataloader = PromptDataLoader(
        dataset = train_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = train_batch_size,
        shuffle = True,
        max_seq_length=max_seq_length,
        truncate_method="head",
    )
    
    val_dataloader = PromptDataLoader(
        dataset = val_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = val_batch_size,
        max_seq_length=max_seq_length,
        truncate_method="head",
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = PromptUtil.verbalizer,
        tokenizer = tokenizer,
    )

    model = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )
    model.to(device)
    
    num_training_steps = num_epochs * len(train_dataloader)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * warmup_ratio), num_training_steps=num_training_steps)
    # lr_scheduler = get_constant_schedule(optimizer=optimizer)
    loss_func = CrossEntropyLoss()
    # loss_func = CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    # loss_func = FocalLoss(alpha=0.6, reduction="mean")
    
    progress_bar = tqdm(range(num_training_steps))
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    early_stop = 0
    
    # num_epochs = num_training_steps // len(train_dataloader)
    train_steps = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = loss_func(logits, batch["label"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
            train_steps += 1
            
            # if train_steps % eval_step == 0:
        model.eval()
        predictions = []
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch)
                total_loss += loss_func(logits, batch["label"])
                preds = torch.argmax(logits, dim = -1)
                predictions += preds.flatten().tolist()

        f1 = metric_f1.compute(predictions=predictions, references=references)["f1"]
        precision = metric_precision.compute(predictions=predictions, references=references)["precision"]
        recall = metric_recall.compute(predictions=predictions, references=references)["recall"]

        logger.info(f"Epoch {epoch}: Loss = {total_loss/len(val_dataloader)}, F1 = {f1}, Precision = {precision}, Recall = {recall}")

        if best_f1 < f1:
            best_f1 = f1
            best_recall = recall
            best_precision = precision
            early_stop = 0
            torch.save(model.state_dict(), f"{exp_name}/{model_name}_{seed}_{lr}_{epoch}_{f1:.2f}_{precision:.2f}_{recall:.2f}.pt")
        else:
            early_stop += 1
            if early_stop >= early_stop_steps:
                logger.info(f"Early stop at epoch {epoch}")
                break
        
    logger.info(f"Final Result: F1 = {best_f1}, Precision = {best_recall}, Recall = {best_precision}")

if __name__ == "__main__":
    logging.init_logger()
    seed = 42
    seed_everything(seed)
    train(seed)
