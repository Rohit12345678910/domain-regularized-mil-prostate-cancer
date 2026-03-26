"""
DINOv2 Feature Extraction Script

This script extracts patch-level features from WSI patches using DINOv2
and saves them as NumPy arrays for MIL training.

Input:
- Patch folders (train/val/test)

Output:
- Feature files (.npy) per slide
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2 = dinov2.to(device)
dinov2.eval()

print("DINOv2 loaded")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Paths
BASE_DIR = "./data"  

PATCH_DIR = os.path.join(BASE_DIR, "patches")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

# Feature Extraction Function
def extract_slide_features(slide_folder):

    patch_files = sorted(os.listdir(slide_folder))
    features = []

    for patch_file in patch_files:

        patch_path = os.path.join(slide_folder, patch_file)

        try:
            with Image.open(patch_path) as img:
                img = img.convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
        except:
            continue

        with torch.no_grad():
            feat = dinov2(img)

        features.append(feat.cpu().numpy())

    if len(features) == 0:
        return None

    return np.vstack(features)


# Main Function
def main():

    splits = ["train", "val", "test"]

    for split in splits:

        print(f"\nProcessing {split}...")

        split_patch_dir = os.path.join(PATCH_DIR, split)
        split_feature_dir = os.path.join(FEATURE_DIR, split)

        # Check if path exists
        if not os.path.exists(split_patch_dir):
            print(f"Path not found: {split_patch_dir}")
            continue

        os.makedirs(split_feature_dir, exist_ok=True)

        # Only include folders (each folder = one slide)
        slides = [
            s for s in os.listdir(split_patch_dir)
            if os.path.isdir(os.path.join(split_patch_dir, s))
        ]

        print(f"Total slides in {split}: {len(slides)}")

        for slide in tqdm(slides):

            slide_folder = os.path.join(split_patch_dir, slide)
            save_path = os.path.join(split_feature_dir, slide + ".npy")

            # Skip if already processed
            if os.path.exists(save_path):
                continue

            feats = extract_slide_features(slide_folder)

            if feats is not None:
                np.save(save_path, feats)


# Entry Point
if __name__ == "__main__":
    main()