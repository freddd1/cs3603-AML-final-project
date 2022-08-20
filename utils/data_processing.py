from typing import List
import pandas as pd
import torch
from torch.utils.data import DataLoader
import itertools
from nltk.corpus import stopwords
from cleantext import clean


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


def prepre_metadata_to_model(path_to_metadata: str, cutoff: int=2) -> pd.DataFrame:
    """
    :param df:
    :param cutoff: is the cutoff of the overall score. It is the threshold we will use in order to create a binary label
    :return:
    """
    df = pd.read_csv(path_to_metadata)
    return_df = pd.DataFrame(columns=['labels', 'id1', 'id2'])

    return_df[['id1', 'id2']] = df.pair_id.str.split('_', 1, expand=True)
    return_df = return_df.astype({'id1': 'int', 'id2': 'int'})

    return_df['labels'] = 0
    return_df['labels'][df['Overall'] >= cutoff] = 1

    return return_df


def prepre_articles_to_model(path_to_data: str, col_text_to_use: str) -> pd.DataFrame:
    '''

    :param path_to_data:
    :param col_text_to_use: from the group {'title', 'text'}
    :return:
    '''

    df = pd.read_csv(path_to_data)
    return df[['news_id', col_text_to_use]].rename(columns={'news_id': 'id', col_text_to_use: 'input_text'})


def _filter_metadata_rows(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Filter only rows that we have their data in self.data
    :return: filter metadata file
    """
    idx = []
    ids_to_check = data.id.tolist()
    for i, (id1, id2) in enumerate(zip(metadata.id1.tolist(), metadata.id2.tolist())):
        if id1 in ids_to_check and id2 in ids_to_check:
            idx.append(i)
    return metadata.loc[idx, :]


def _merge_data_metadata(data, metadata):
    m = metadata.merge(data, left_on='id1', right_on='id', how='left').rename(columns={'input_text': 'text1'})
    m = m.merge(data, left_on='id2', right_on='id', how='left').rename(columns={'input_text': 'text2'})
    m = m.drop(columns=[col for col in m.columns if col.startswith('id')])
    return m


def clean_text(s):
    # if not isinstance(s, str):
    #     print(s)
    # 1489983888
    if s and isinstance(s, str):
        s = s.lower()
        s = clean(text=s)
        s = " ".join([word for word in s.split(' ') if word not in stopwords.words('english')])
        return s
    return ''


def prepre_data_to_model(path_to_metadata: str,
                         path_to_data: str,
                         cutoff: int = 2,
                         col_text_to_use: str = 'title'
                         ):
    """
    :param path_to_metadata:
    :param path_to_data:
    :param cutoff:
    :param col_text_to_use: from the group {'title', 'text'}
    :return:
    """

    data = prepre_articles_to_model(path_to_data, col_text_to_use)
    metadata = prepre_metadata_to_model(path_to_metadata, cutoff=cutoff)
    metadata = _filter_metadata_rows(data, metadata)
    combined_df = _merge_data_metadata(data, metadata)
    for col in ['text1', 'text2']:
        combined_df.loc[:, col] = combined_df[col].apply(clean_text)
    return combined_df
