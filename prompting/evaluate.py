import sys

from git import Reference
sys.path.append('..')
import logger as logging
from logger import logger
import torch
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from dont_patronize_me import DontPatronizeMe
from tqdm import tqdm
from prompting.util import PromptUtil, load_references, metric_f1, metric_precision, metric_recall

model_path = "prompt-tuning-paraphrases/bart_42_1e-05_2_0.60_0.55_0.66.pt"

logging.init_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
classes = [0, 1]

plm, tokenizer, model_config, WrapperClass = load_plm("bart", "facebook/bart-large")

promptTemplate = ManualTemplate(
    text = PromptUtil.template,
    tokenizer = tokenizer,
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

model.load_state_dict(torch.load(model_path))

model.to(device)


val_dpm = DontPatronizeMe('../data', '')
val_dpm.load_task1("test_long.tsv")
val_df = val_dpm.train_task1_df
# val_df = val_df[val_df.orig_label=="2"]

val_dataset = PromptUtil.data2examples(val_df)

references = load_references(val_dataset)

dataloader = PromptDataLoader(
    dataset = val_dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size = 32,
    max_seq_length = 128
)

model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        preds = torch.argmax(logits, dim = -1)
        predictions += preds.flatten().tolist()
    
        
f1 = metric_f1.compute(predictions=predictions, references=references)["f1"]
precision = metric_precision.compute(predictions=predictions, references=references)["precision"]
recall = metric_recall.compute(predictions=predictions, references=references)["recall"]
        
logger.info(f"F1: {f1}, Precision: {precision}, Recall: {recall}")