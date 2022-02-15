from data_loader import Dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForSequenceClassification
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large")

training_data = Dataset(train=True)
test_data = Dataset(train=False)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

