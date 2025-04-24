import torch
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from nn_util import SeqNN
from torch.utils.data import DataLoader
from plot_results import plot_loss
from new_dataset import SequenceDataset, collate_fn, OHEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Hyperparameters
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
LEARNING_RATE = 1e-3
ALPHA = 1.0  # weight for regression loss


def train_and_evaluate(path='./data/train_data.tsv'):
    # Load the dataset
    df = pd.read_csv(path, sep='\t')
    print(f"Loaded dataset with {len(df)} sequences.")

    results = {}
    oh_encoder = OHEncoder()

    # Perform 5-fold cross-validation
    for fold in range(5):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # Split data into training (80%) and validation (20%) sets
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=fold, stratify=df['insert_chrom']
        )

        # Create datasets and data loaders
        train_ds = SequenceDataset(train_df, oh_encoder)
        val_ds = SequenceDataset(val_df, oh_encoder)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Initialize the model, optimizer, scheduler, and loss functions
        model = SeqNN(seqsize=271, use_single_channel=True).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion_cls = nn.BCELoss()
        criterion_reg = nn.MSELoss()

        best_f1 = 0.0
        best_model_state = None
        train_losses, val_losses = [], []

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}")
            for batch in pbar:
                if batch is None:
                    continue
                raw_seqs, oh_tensor, y_cls, y_reg = batch
                oh_tensor, y_cls, y_reg = oh_tensor.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)

                optimizer.zero_grad()
                x = torch.cat([oh_tensor.squeeze(1), torch.zeros((oh_tensor.size(0), 2, oh_tensor.size(-1)), device=DEVICE)], dim=1)
                out_cls, out_reg = model(x)
                loss_cls = criterion_cls(out_cls.squeeze(-1), y_cls)
                loss_reg = criterion_reg(out_reg.squeeze(-1), y_reg)
                loss = loss_cls + ALPHA * loss_reg
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                pbar.set_postfix(loss=loss.item(), loss_cls=loss_cls.item(), loss_reg=loss_reg.item())

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation loop
            model.eval()
            epoch_val_loss = 0
            total_f1 = total_prec = total_rec = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    raw_seqs, oh_tensor, y_cls, y_reg = batch
                    oh_tensor, y_cls, y_reg = oh_tensor.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)
                    x = torch.cat([oh_tensor.squeeze(1), torch.zeros((oh_tensor.size(0), 2, oh_tensor.size(-1)), device=DEVICE)], dim=1)
                    out_cls, out_reg = model(x)
                    loss_cls = criterion_cls(out_cls.squeeze(-1), y_cls)
                    loss_reg = criterion_reg(out_reg.squeeze(-1), y_reg)
                    val_loss = loss_cls + ALPHA * loss_reg
                    epoch_val_loss += val_loss.item()

                    preds = (out_cls.squeeze(-1) > 0.5).long().cpu().numpy()
                    y_true = y_cls.cpu().long().numpy()
                    total_f1 += f1_score(y_true, preds)
                    total_prec += precision_score(y_true, preds)
                    total_rec += recall_score(y_true, preds)

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            f1 = total_f1 / len(val_loader)
            prec = total_prec / len(val_loader)
            rec = total_rec / len(val_loader)

            scheduler.step(avg_val_loss)
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

            # Save the model if it achieves the best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()

        print(f"Best F1 score for Fold {fold+1}: {best_f1:.4f}")
        results[f"fold_{fold+1}"] = {'train_losses': train_losses, 'val_losses': val_losses, 'best_f1': best_f1}

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')
        print("Best model saved as 'best_model.pth'.")

    # Save cross-validation results
    torch.save(results, 'cv_results.pt')
    print("Cross-validation results saved as 'cv_results.pt'.")


def example_train(path='./data/train_data.tsv', val_chroms=['chr10', 'chr11']):
    print("STARTING TRAINING")

    df = pd.read_csv(path, sep='\t')
    train_df = df[~df['insert_chrom'].isin(val_chroms)]
    val_df = df[df['insert_chrom'].isin(val_chroms)]

    oh_encoder = OHEncoder()
    train_dataset = SequenceDataset(train_df, oh_encoder)
    val_dataset = SequenceDataset(val_df, oh_encoder)
    # Length of sequences is 271, so we can use a fixed size for the model
    model = SeqNN(seqsize=271, use_single_channel=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            if batch is None:
                continue
            raw_seqs, oh_tensor, y_cls, y_reg = batch
            oh_tensor, y_cls, y_reg = oh_tensor.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)

            optimizer.zero_grad()
            x = torch.cat([oh_tensor.squeeze(1), torch.zeros((oh_tensor.shape[0], 2, oh_tensor.shape[-1]), device=DEVICE)], dim=1)
            out_cls, out_reg = model(x)
            loss_reg = criterion_reg(out_reg.squeeze(-1), y_reg)
            loss_cls = criterion_cls(out_cls.squeeze(-1), y_cls)
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss_reg=loss_reg.item(), loss_cls=loss_cls.item(), loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Walidacja
        model.eval()
        epoch_val_loss = 0
        avg_reg = 0
        avg_cls = 0
        f1 = 0
        precision = 0
        recall = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                raw_seqs, oh_tensor, y_cls, y_reg = batch
                oh_tensor, y_cls, y_reg = oh_tensor.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)
                x = torch.cat([oh_tensor.squeeze(1), torch.zeros((oh_tensor.shape[0], 2, oh_tensor.shape[-1]), device=DEVICE)], dim=1)
                out_cls, out_reg = model(x)
                loss_reg = criterion_reg(out_reg.squeeze(-1), y_reg)
                avg_reg += loss_reg
                loss_cls = criterion_cls(out_cls.squeeze(-1), y_cls)
                avg_cls += loss_cls
                loss_val = loss_cls + loss_reg
                epoch_val_loss += loss_val.item()
                f1 += f1_score(y_cls.cpu().numpy(), (out_cls.squeeze(-1).cpu().numpy() > 0.5).astype(int))
                precision += precision_score(y_cls.cpu().numpy(), (out_cls.squeeze(-1).cpu().numpy() > 0.5).astype(int))
                recall += recall_score(y_cls.cpu().numpy(), (out_cls.squeeze(-1).cpu().numpy() > 0.5).astype(int))
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_reg /= len(val_loader)
        avg_cls /= len(val_loader)
        f1 /= len(val_loader)
        precision /= len(val_loader)
        recall /= len(val_loader)
        val_losses.append(avg_val_loss)
        print(avg_val_loss, f1)


    scheduler.step(avg_val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Reg Loss = {avg_reg:.4f}, Cls Loss = {avg_cls:.4f}, F1 = {f1:.4f}, Precision = {precision}, Recall = {recall}")

    plot_loss(train_losses, val_losses)

if __name__ == '__main__':
    # example_train()
    train_and_evaluate()
    print("Training and evaluation completed.")
