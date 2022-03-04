import sys
sys.path.append('..')
import torch


class PCLPromptDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset="train"):
        # get data
        sources, targets = [], []
        with open(f"../data/prompts_{dataset}.txt", 'r') as f:
            for line in f:
                line_seg = line.strip().split('\t')
                sources.append(line_seg[0])
                targets.append(line_seg[1])
        
        self.sources = tokenizer(sources, truncation=True, padding=True, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            self.targets = tokenizer(targets, truncation=True, padding=True, return_tensors="pt")
        
    def __len__(self):
        return len(self.sources["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.sources.items()}
        label = {key: val[idx].clone().detach() for key, val in self.targets.items()}
        item["labels"] = label["input_ids"]
        return item
    
class PCLPromptTestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        # get data
        sources = []
        with open(f"../data/task4_test_prompt.txt", 'r') as f:
            for line in f:
                sources.append(line.strip())
        
        self.sources = tokenizer(sources, truncation=True, padding=True, return_tensors="pt")
        
    def __len__(self):
        return len(self.sources["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.sources.items()}
        return item