import torch
from torch.utils.data import DataLoader
from typing import List
import itertools


def stuck_batch_input_ids(dl: DataLoader) -> torch.Tensor:
    all_ids = [batch['input_ids'] for batch in dl]
    return torch.cat(all_ids, axis=0)


def create_sentences_corpus(dl: DataLoader) -> List:
    sentences = []
    sentences.extend([batch['sentences1'] for batch in dl])
    sentences.extend([batch['sentences2'] for batch in dl])
    return list(itertools.chain(*sentences))
