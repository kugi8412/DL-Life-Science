import csv
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from nn_util import SeqNN
from new_dataset import SequenceDataset, OHEncoder, collate_fn


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
    print("CUDA available:", torch.cuda.is_available())
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
    output_file = "predictions.tsv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["id", "cls", "reg"])
        for i, (y_cls, y_reg) in enumerate(zip(pred_cls, pred_reg)):
            writer.writerow([i, y_cls[0], y_reg[0]])
    
    print(f"Predictions saved to {output_file}")
