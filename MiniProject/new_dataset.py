import torch
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder as Encoder


class OHEncoder:
    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse_output=False, categories=[categories])
        self.dictionary = categories
        self.encoder.fit(categories.reshape(-1, 1))

    def __call__(self, seq, info=False):
        seq = list(seq)
        # For random nucleotide -  'N' we choose random instead
        if 'N' in seq:
            pos = [i for i, el in enumerate(seq) if el == 'N']
            for p in pos:
                seq[p] = random.choice(self.dictionary)

        s = np.array(seq).reshape(-1, 1)
        encoded = self.encoder.transform(s).T  # result (4, seq_len)
        return torch.tensor(encoded).unsqueeze(0).unsqueeze(0).float()
    

class SequenceDataset(Dataset):
    def __init__(self, df, oh_encoder, test=False):
        """ DataFrame must have columns:
          - sequence (with nucleotide sequence)
          - rna_dna_ratio (continous value, float)
          - is_active (0 or 1, float)
        """
        self.df = df.reset_index(drop=True)
        self.oh_encoder = oh_encoder
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence']
        oh = self.oh_encoder(seq)  # shape (1, 4, seq_len)
        if oh is None:
            return None
        if self.test:
            return seq, oh
        label_cls = float(row['is_active'])
        label_reg = float(row['rna_dna_ratio'])
        return seq, oh, torch.tensor(label_cls, dtype=torch.float32), torch.tensor(label_reg, dtype=torch.float32)

def collate_fn(batch):
    """ Function to collate a batch of data from the dataset.
    It removes None values and stacks the tensors.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    elif len(batch[0]) == 2:
        raw_seqs, oh_list = zip(*batch)
        oh_tensor = torch.cat(oh_list, dim=0)
        return list(raw_seqs), oh_tensor
    raw_seqs, oh_list, y_cls, y_reg = zip(*batch)
    oh_tensor = torch.cat(oh_list, dim=0)  # shape: (B, 1, 4, seq_len)
    y_cls = torch.stack(y_cls)
    y_reg = torch.stack(y_reg)
    return list(raw_seqs), oh_tensor, y_cls, y_reg
