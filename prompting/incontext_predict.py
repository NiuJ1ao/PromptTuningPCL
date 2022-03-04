import sys
sys.path.append('..')
import logger as logging
import torch
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from dont_patronize_me import DontPatronizeMe
from tqdm import tqdm
from prompting.util import IncontextUtil, load_references
from datasets import load_metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process_data():
    classes = [0, 1]

    plm, tokenizer, model_config, WrapperClass = load_plm("bart", "facebook/bart-large")

    promptTemplate = ManualTemplate(
        text = IncontextUtil.template,
        tokenizer = tokenizer,
    )

    train_dpm = DontPatronizeMe('../data', '../data/task4_test.tsv')
    train_dpm.load_task1("train.tsv")
    train_df = train_dpm.train_task1_df
    pos_df = train_df[train_df.label==1]
    neg_df = train_df[train_df.label==0]
    
    test_dpm = DontPatronizeMe('../data', '../data/task4_test.tsv') 
    test_dpm.load_test()
    test_df = test_dpm.test_set_df
    test_dataset = IncontextUtil.test2examples(test_df, pos_df, neg_df)
    references = None

    test_dataloader = PromptDataLoader(
        dataset = test_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size = 32,
        max_seq_length = 512
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = IncontextUtil.verbalizer,
        tokenizer = tokenizer,
    )
    
    return plm, promptTemplate, promptVerbalizer, test_dataloader, references

def predict(model_path, plm, promptTemplate, promptVerbalizer, dataloader):
    model = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    logits_list = torch.tensor([], device=device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch) # batch_size x num_classes
            logits_list = torch.cat([logits_list, logits], dim=0)

    assert logits_list.shape[1] == 2
    return logits_list

def evaluate(logits, references):
    metric_f1 = load_metric("f1")
    metric_precision = load_metric("precision")
    metric_recall = load_metric("recall")
                
    predictions = ensemble(logits)

    f1 = metric_f1.compute(predictions=predictions, references=references)["f1"]
    precision = metric_precision.compute(predictions=predictions, references=references)["precision"]
    recall = metric_recall.compute(predictions=predictions, references=references)["recall"]
                
    print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")
    
def ensemble(logits):
    # uniform
    probs = logits.softmax(-1).mean(dim=0)
    return probs.argmax(dim=-1).flatten().tolist()
    
if __name__ == "__main__":
    logging.init_logger()
    paths = [
        # "fewshot-1000-demonstrator-5epochs/bart_13_1e-05_6_0.55_0.45_0.70.pt",
        # "fewshot-1000-demonstrator-5epochs/bart_21_1e-05_8_0.53_0.41_0.74.pt",
        # "fewshot-1000-demonstrator-5epochs/bart_42_1e-05_9_0.51_0.43_0.63.pt",
        # "fewshot-1000-demonstrator-5epochs/bart_87_1e-05_6_0.50_0.40_0.66.pt",
        # "fewshot-1000-demonstrator-5epochs/bart_100_1e-05_4_0.52_0.41_0.71.pt",
        
        "fewshot-4000-demonstrator-5epochs-1/bart_13_1e-05_8_0.56_0.48_0.68.pt",
        "fewshot-4000-demonstrator-5epochs-1/bart_21_1e-05_5_0.59_0.52_0.67.pt",
        "fewshot-4000-demonstrator-5epochs-1/bart_42_1e-05_8_0.56_0.46_0.73.pt",
        "fewshot-4000-demonstrator-5epochs-1/bart_87_1e-05_9_0.52_0.43_0.66.pt",
        "fewshot-4000-demonstrator-5epochs-1/bart_100_1e-05_0_0.44_0.56_0.37.pt",
    ]
    
    plm, promptTemplate, promptVerbalizer, dataloader, references = process_data()
    
    logits = []
    for path in paths:
        logits.append(predict(path, plm, promptTemplate, promptVerbalizer, dataloader))
    logits = torch.stack(logits)
    
    predictions = ensemble(logits)
    with open("task1.txt", "w") as outf:
        for p in predictions:
            outf.write(str(p)+"\n")