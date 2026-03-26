"""
MIL Training Script using DINOv2 Features

This script trains MIL models (CLAM and TransMIL) on extracted DINOv2 features
and evaluates performance on in-domain (Radboud) and cross-domain (Karolinska).

Input:
- Feature files (.npy)
- CSV files with labels

Output:
- Model checkpoints
- Evaluation metrics (AUC, Accuracy)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


# CONFIG
BASE_DIR = "./data"

TRAIN_FEATURES = os.path.join(BASE_DIR, "features/train")
VAL_FEATURES   = os.path.join(BASE_DIR, "features/val")
TEST_FEATURES  = os.path.join(BASE_DIR, "features/test")

TRAIN_CSV = os.path.join(BASE_DIR, "splits/train.csv")
VAL_CSV   = os.path.join(BASE_DIR, "splits/val.csv")
TEST_CSV  = os.path.join(BASE_DIR, "splits/test.csv")

KARO_FEATURES = os.path.join(BASE_DIR, "features/karolinska")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)



# DATASET

class WSIDataset(Dataset):

    def __init__(self, feature_dir, csv_file):

        self.feature_dir = feature_dir
        self.df = pd.read_csv(csv_file)

        # keep only existing features
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



# MODELS

# CLAM Model
class CLAM(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512, n_classes=2):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):

        h = self.fc(x)
        A = torch.softmax(self.attention(h), dim=0)
        M = torch.sum(A * h, dim=0)

        return self.classifier(M).unsqueeze(0)


# TransMIL Model
class TransMIL(nn.Module):

    def __init__(self, dim=768, num_classes=2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):

        x = x.unsqueeze(0)
        cls_token = self.cls_token.expand(1, -1, -1)

        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer(x)

        return self.classifier(x[:, 0])


# TRAINING

def train_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for features, label in loader:

        features = features.squeeze(0).to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(features)

        loss = criterion(output, label.unsqueeze(0))
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

            output = model(features)
            prob = torch.softmax(output, dim=1)[:, 1]

            preds.append(prob.cpu().numpy())
            labels.append(label.numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds > 0.5)

    return auc, acc


# MAIN

def main():

    # Load data
    train_loader = DataLoader(WSIDataset(TRAIN_FEATURES, TRAIN_CSV), batch_size=1, shuffle=True)
    val_loader   = DataLoader(WSIDataset(VAL_FEATURES, VAL_CSV), batch_size=1)
    test_loader  = DataLoader(WSIDataset(TEST_FEATURES, TEST_CSV), batch_size=1)

    print("Datasets loaded")

    # Choose model
    model = CLAM().to(DEVICE)
    # model = TransMIL().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 20

    for epoch in range(EPOCHS):

        loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auc, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    # Final evaluation
    test_auc, test_acc = evaluate(model, test_loader)

    print("\nFinal Results (Radboud)")
    print("AUC:", test_auc)
    print("Accuracy:", test_acc)


if __name__ == "__main__":
    main()