from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
)
from data import *
import torch
from sklearn.preprocessing import LabelEncoder


class RelationExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(torch.int64) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]]).to(torch.int64)
        return item

    def __len__(self):
        return len(self.labels)