import sys
sys.path.append('..')
import logger as logging
from logger import logger
import torch
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from dont_patronize_me import DontPatronizeMe
from tqdm import tqdm
from prompting.util import PromptUtil

model_path = "prompt-tuning-paraphrases/bart_42_1e-05_2_0.60_0.55_0.66.pt"

logging.init_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
classes = [0, 1]

plm, tokenizer, model_config, WrapperClass = load_plm("bart", "facebook/bart-large")

promptTemplate = ManualTemplate(
    text = PromptUtil.template,
    tokenizer = tokenizer,
)

dpm = DontPatronizeMe('../data', '../data/task4_test.tsv')
dpm.load_test()
test_dataset = PromptUtil.test2examples(dpm.test_set_df)

dataloader = PromptDataLoader(
    dataset = test_dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size = 32,
    max_seq_length = 128
)

# dpm = DontPatronizeMe('../data', '../data/task4_test.tsv')
# train_ids, test_ids = load_paragraph_ids("../data")
# dpm.load_task1()
# val_df = rebuild_raw_dataset(dpm.train_task1_df, test_ids)
# val_dataset = PromptUtil.data2examples(val_df)

# dataloader = PromptDataLoader(
#     dataset = val_dataset,
#     tokenizer = tokenizer,
#     template = promptTemplate,
#     tokenizer_wrapper_class=WrapperClass,
#     batch_size = 4,
#     max_seq_length=1024
# )

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
model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        preds = torch.argmax(logits, dim = -1)
        predictions += preds.flatten().tolist()

with open("task1.txt", "w") as outf:
    for p in predictions:
        outf.write(str(p)+"\n")