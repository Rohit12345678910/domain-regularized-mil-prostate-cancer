"""
Domain-Regularized MIL Training Script

This script implements the proposed method:
Domain-Regularized Multiple Instance Learning (MIL)

Key Idea:
- Standard MIL attention is unstable under domain shift
- We regularize attention distribution to improve generalization

Loss:
    total_loss = classification_loss + lambda * attention_regularization

Regularization:
- Entropy - stabilize attention
- Sparsity - focus on important patches

Output:
- Model checkpoints
- Predictions for evaluation
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

        # Keep only valid samples
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


# MODEL (PROPOSED)

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

        # Attention scores
        A = self.attention(x)
        A = torch.softmax(A, dim=0)

        # Aggregation
        M = torch.sum(A * x, dim=0)

        logits = self.classifier(M)

        return logits.unsqueeze(0), A


# REGULARIZATION (CORE NOVELTY)

def attention_regularization(A):
    """
    Regularization on attention weights

    - Entropy - smooth/stable attention
    - Sparsity - focus on key patches
    """

    entropy = -torch.sum(A * torch.log(A + 1e-8))
    sparsity = torch.sum(torch.abs(A))

    return -entropy + 0.001 * sparsity


# TRAINING

def train_epoch(model, loader, optimizer, criterion, lambda_reg):

    model.train()
    total_loss = 0

    for features, label in loader:

        features = features.squeeze(0).to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        logits, A = model(features)

        cls_loss = criterion(logits, label.unsqueeze(0))
        reg_loss = attention_regularization(A)

        loss = cls_loss + lambda_reg * reg_loss

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

    return auc, acc, preds, labels


# MAIN

def main():

    # Load datasets
    train_loader = DataLoader(MILDataset(TRAIN_FEATURES, TRAIN_CSV), batch_size=1, shuffle=True)
    val_loader   = DataLoader(MILDataset(VAL_FEATURES, VAL_CSV), batch_size=1)
    test_loader  = DataLoader(MILDataset(TEST_FEATURES, TEST_CSV), batch_size=1)

    print("Datasets loaded")

    # Model
    model = DomainRegularizedMIL().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    lambda_reg = 0.02
    EPOCHS = 20

    # Training loop
    for epoch in range(EPOCHS):

        loss = train_epoch(model, train_loader, optimizer, criterion, lambda_reg)

        val_auc, val_acc, _, _ = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Final evaluation
    test_auc, test_acc, preds, labels = evaluate(model, test_loader)

    print("\nFinal Results (Radboud)")
    print("AUC:", test_auc)
    print("Accuracy:", test_acc)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/domain_regularized_mil.pth")

    print("Model saved")

    # Save predictions
    os.makedirs("results", exist_ok=True)

    np.save("results/preds.npy", preds)
    np.save("results/labels.npy", labels)

    print("Results saved")


if __name__ == "__main__":
    main()