from torch.utils.data import Dataset
import transformers
from typing import List, Tuple


class SentencesDataset(Dataset):
    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int]):
        self.labels = labels
        self.sentences1 = sentences1
        self.sentences2 = sentences2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> dict:
        s1 = self.sentences1[idx]
        s2 = self.sentences2[idx]
        label = self.labels[idx]
        return dict(sentences1=s1, sentences2=s2, labels=label)


class TokenizedDataset(Dataset):
    def __init__(self, batch: transformers.tokenization_utils_base.BatchEncoding, labels: List[int]):
        self.labels = labels
        self.input_ids = batch['input_ids']
        self.token_type_ids = batch['token_type_ids']
        self.attention_mask = batch['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]

        return dict(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels)
