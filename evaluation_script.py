import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder as Encoder
from tltorch import TRL
from collections import OrderedDict

class OHEncoder:
    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse_output=False, categories=[categories])
        self.dictionary = categories
        self.encoder.fit(categories.reshape(-1, 1))

    def __call__(self, seq, info=False):
        seq = list(seq)
        # Jeśli w sekwencji występuje znak 'N', w naszych danych nie ma, ale może być
        if 'N' in seq:
            pos = [i for i, el in enumerate(seq) if el == 'N']
            for p in pos:
                seq[p] = random.choice(self.dictionary)

        s = np.array(seq).reshape(-1, 1)
        encoded = self.encoder.transform(s).T  # wynik: macierz (4, seq_len)
        return torch.tensor(encoded).unsqueeze(0).unsqueeze(0).float()
    

class SequenceDataset(Dataset):
    def __init__(self, df, oh_encoder):
        """ df powinno zawierać kolumny:
          - sequence
          - rna_dna_ratio
          - is_active
        """
        self.df = df.reset_index(drop=True)
        self.oh_encoder = oh_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence']  # sekwencja jako str
        oh = self.oh_encoder(seq)  # tensor (1, 4, seq_len)
        if oh is None:
            return None
        label_cls = float(row['is_active'])
        label_reg = float(row['rna_dna_ratio'])
        return seq, oh, torch.tensor(label_cls, dtype=torch.float32), torch.tensor(label_reg, dtype=torch.float32)

def collate_fn(batch):
    # Usuwamy próbki None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    raw_seqs, oh_list, y_cls, y_reg = zip(*batch)
    oh_tensor = torch.cat(oh_list, dim=0)  # kształt: (B, 1, 4, seq_len)
    y_cls = torch.stack(y_cls)
    y_reg = torch.stack(y_reg)
    return list(raw_seqs), oh_tensor, y_cls, y_reg


class Bilinear(nn.Module):
    """
    Bilinear layer introduces pairwise product to a NN to model possible combinatorial effects.
    This particular implementation attempts to leverage the number of parameters via low-rank tensor decompositions.

    Parameters
    ----------
    n : int
        Number of input features.
    out : int, optional
        Number of output features. If None, assumed to be equal to the number of input features. The default is None.
    rank : float, optional
        Fraction of maximal to rank to be used in tensor decomposition. The default is 0.05.
    bias : bool, optional
        If True, bias is used. The default is False.

    """
    def __init__(self, n: int, out=None, rank=0.05, bias=False):
        super().__init__()
        if out is None:
            out = (n, )
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))

class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension.

    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.

    """
    def __init__(self, module: nn.Module, dim=-1):
        super().__init__()
        self.mod = module
        self.dim = dim

    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)

class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.

    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.

    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SeqNN(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')

    def __init__(self,
                seqsize,
                use_single_channel,
                block_sizes=[256, 256, 128, 128, 64, 64, 32, 32],
                ks=5,
                resize_factor=4,
                activation=nn.SiLU,
                filter_per_group=2,
                se_reduction=4,
                final_ch=18,
                bn_momentum=0.1,
                to_extract=[], # indeksy warstw (oprócz ostatniej), których wyjścia wchodzą do klasyfikatora
                pool_dim=4,
                standalone=True):
        super().__init__()
        self.to_extract = to_extract
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.use_single_channel = use_single_channel
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        self.pool_dim = pool_dim
        self.standalone = standalone
        self.class_head = nn.Sequential(
            nn.Linear((self.block_sizes[-1] + sum([block_sizes[i+1] for i in self.to_extract])) * (self.pool_dim), 1),
            # nn.Linear(944,1),
            nn.Sigmoid()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.final_ch * self.seqsize, 1),
        )
        seqextblocks = OrderedDict()

        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=6 if self.use_single_channel else 5,
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0],
                                     momentum=self.bn_momentum),
                       activation()#Exponential(block_sizes[0]) #activation()
        )
        seqextblocks[f'blc0'] = block


        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                        #nn.Dropout(0.1),
                        nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor,
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz * self.resize_factor), #activation(),


                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor,
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz * self.resize_factor), #activation(),
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                    #    nn.Dropout(0.1),
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz,
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz), #activation(),

            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                             padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz,
                                      momentum=self.bn_momentum),
                       activation(),#Exponential(sz), #activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block



        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper =  block = nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )

        self.register_buffer('bins', torch.arange(start=0, end=18, step=1, requires_grad=False))

    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        res_extract = []

        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
            if i in self.to_extract:
                res_extract.append(x)
        return x, res_extract

    def forward(self, x):
        f, ext = self.feature_extractor(x)
        # print([a.shape for a in ext])
        x = self.mapper(f).flatten(1) # w publikacji robili coś takiego, że wyjście na wyjściu z tej warstwy robili pooling do dim=1 i to traktowali jako logity do pseudo-klasyfikacji,
        #                      ja to wyrzuciłem i jest zwykła warstwa liniowa
        f = F.adaptive_max_pool1d(f, self.pool_dim).flatten(1) # poole potrzebne bo jest potężny overfitting
        ext_pooled = [F.adaptive_max_pool1d(e, self.pool_dim).flatten(1) for e in ext]
        f = torch.cat([f] + ext_pooled, dim=1)
        # f = torch.cat([e.flatten(1) for e in ext], dim=1)
        if self.standalone:
            out_class = self.class_head(f.flatten(1))
            score = self.reg_head(x)
            return out_class, score # class,reg
        return f,x # reg,class

def load_data(dataset):
    df = pd.read_csv(dataset, sep='\t')
    oh_encoder = OHEncoder()
    dataset = SequenceDataset(df, oh_encoder)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    return loader

def eval(loader, model):
    y_cls = []
    y_reg = []
    model.eval()
    with torch.no_grad():
        for raw_seqs, oh_tensor, _, _ in loader:
            oh_tensor = oh_tensor.to(DEVICE)
            x = torch.cat([oh_tensor.squeeze(1), torch.zeros((oh_tensor.shape[0], 1, oh_tensor.shape[-1]), device=DEVICE), torch.zeros((oh_tensor.shape[0], 1, oh_tensor.shape[-1]), device=DEVICE)], dim=1)
            y_cls_pred, y_reg_pred = model(x)
            y_cls.append(y_cls_pred.cpu().numpy())
            y_reg.append(y_reg_pred.cpu().numpy())
    y_cls = np.concatenate(y_cls)
    y_cls = (y_cls>0.5).astype(np.int32)
    y_reg = np.concatenate(y_reg)
    return y_cls, y_reg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqNN(
        seqsize=271,
        use_single_channel=True,
        block_sizes= [256, 128, 128, 64, 64, 64, 64],
        ks=7,
        resize_factor=4,
        se_reduction=4,
        final_ch=18,
        to_extract = [],
        pool_dim=8,
        standalone=True
        ).to(DEVICE)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    loader = load_data(args.dataset)
    pred_cls, pred_reg = eval(loader, model)
    print("id\tcls\treg")
    for i, (y_cls, y_reg) in enumerate(zip(pred_cls, pred_reg)):
        print(f"{i}\t{y_cls[0]}\t{y_reg[0]}")
    # print(f1_score(loader.dataset.df.is_active, pred_cls)) # test czy kolejność się nie zmieniła
    