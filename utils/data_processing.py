from typing import List
import pandas as pd
import torch
from torch.utils.data import DataLoader
import itertools


def load_msrp_txt(file_name: str):
    cols_rename = {'Quality': 'label', '#1 ID': 'id1', '#2 ID': 'id2', '#1 String': 's1', '#2 String': 's2'}
    df = pd.read_csv(f'data/MSRP/{file_name}', sep='\t', error_bad_lines=False).rename(columns=cols_rename)
    df.s1 = df.s1.astype(str)
    df.s2 = df.s2.astype(str)
    return df


def stuck_batch_input_ids(dl: DataLoader) -> torch.Tensor:
    all_ids = [batch['input_ids'] for batch in dl]
    return torch.cat(all_ids, axis=0)


def create_sentences_corpus(dl: DataLoader) -> List:
    sentences = []
    sentences.extend([batch['sentences1'] for batch in dl])
    sentences.extend([batch['sentences2'] for batch in dl])
    return list(itertools.chain(*sentences))
