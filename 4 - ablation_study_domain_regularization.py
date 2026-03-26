"""
Ablation Study Script

Purpose:
To verify that the performance improvement comes from
the proposed attention regularization.

Comparison:
1. Domain-Regularized MIL (proposed)
2. MIL without regularization (ablation)

Only difference:
    With reg: loss = cls_loss + lambda * reg_loss
    Without reg: loss = cls_loss
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


# CONFIG

BASE_DIR = "./data"

TRAIN_FEATURES = os.path.join(BASE_DIR, "features/train")
VAL_FEATURES   = os.path.join(BASE_DIR, "features/val")
TEST_FEATURES  = os.path.join(BASE_DIR, "features/test")

KARO_FEATURES  = os.path.join(BASE_DIR, "features/karolinska")

TRAIN_CSV = os.path.join(BASE_DIR, "splits/train.csv")
VAL_CSV   = os.path.join(BASE_DIR, "splits/val.csv")
TEST_CSV  = os.path.join(BASE_DIR, "splits/test.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# DATASET

class MILDataset(Dataset):

    def __init__(self, feature_dir, csv_file):

        self.feature_dir = feature_dir
        self.df = pd.read_csv(csv_file)

        self.df = self.df[self.df["image_id"].apply(
            lambda x: os.path.exists(os.path.join(feature_dir, x + ".npy"))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        slide_id = row["image_id"]
        label = 0 if row["isup_grade"] <= 1 else 1

        features = np.load(os.path.join(self.feature_dir, slide_id + ".npy"))

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label)


# MODEL (same as proposed)

class DomainRegularizedMIL(nn.Module):

    def __init__(self, input_dim=768):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):

        A = self.attention(x)
        A = torch.softmax(A, dim=0)

        M = torch.sum(A * x, dim=0)

        logits = self.classifier(M)

        return logits.unsqueeze(0), A



# REGULARIZATION

def attention_regularization(A):

    entropy = -torch.sum(A * torch.log(A + 1e-8))
    sparsity = torch.sum(torch.abs(A))

    return -entropy + 0.001 * sparsity


# TRAINING

def train_epoch(model, loader, optimizer, criterion, use_reg, lambda_reg):

    model.train()
    total_loss = 0

    for features, label in loader:

        features = features.squeeze(0).to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        logits, A = model(features)

        cls_loss = criterion(logits, label.unsqueeze(0))

        if use_reg:
            reg_loss = attention_regularization(A)
            loss = cls_loss + lambda_reg * reg_loss
        else:
            loss = cls_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):

    model.eval()

    preds, labels = [], []

    with torch.no_grad():

        for features, label in loader:

            features = features.squeeze(0).to(DEVICE)

            logits, _ = model(features)
            prob = torch.softmax(logits, dim=1)[:, 1]

            preds.append(prob.cpu().numpy())
            labels.append(label.numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)

    return auc, acc


# MAIN

def run_experiment(use_reg, name):

    print(f"\nRunning: {name}")

    train_loader = DataLoader(MILDataset(TRAIN_FEATURES, TRAIN_CSV), batch_size=1, shuffle=True)
    val_loader   = DataLoader(MILDataset(VAL_FEATURES, VAL_CSV), batch_size=1)
    test_loader  = DataLoader(MILDataset(TEST_FEATURES, TEST_CSV), batch_size=1)

    model = DomainRegularizedMIL().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    lambda_reg = 0.02
    EPOCHS = 20

    for epoch in range(EPOCHS):

        loss = train_epoch(model, train_loader, optimizer, criterion, use_reg, lambda_reg)
        val_auc, _ = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    test_auc, test_acc = evaluate(model, test_loader)

    print(f"{name} - Radboud AUC: {test_auc:.4f}")
    print(f"{name} - Accuracy: {test_acc:.4f}")

    return test_auc


def main():

    auc_with_reg = run_experiment(use_reg=True, name="With Regularization")
    auc_without_reg = run_experiment(use_reg=False, name="Without Regularization")

    print("\n===== Final Comparison =====")
    print(f"With Regularization AUC: {auc_with_reg:.4f}")
    print(f"Without Regularization AUC: {auc_without_reg:.4f}")


if __name__ == "__main__":
    main()