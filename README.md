# domain-regularized-mil-prostate-cancer
Domain-Regularized Multiple Instance Learning for Prostate Cancer Detection with DINOv2 features under cross-institution domain shift

# Domain-Regularized MIL for Prostate Cancer Detection
This repository presents a deep learning framework for **prostate cancer detection from Whole Slide Images (WSIs)** using **Multiple Instance Learning (MIL)** and **self-supervised DINOv2 features**.

The primary focus of this work is to address **domain shift** across institutions and improve model generalization.


# Key Contributions

- Comprehensive study on **cross-institution generalization** using the PANDA dataset  
- Use of **DINOv2 self-supervised features** for robust representation learning  
- Proposed **Domain-Regularized MIL framework** to stabilize attention under domain shift  
- Demonstrated significant improvement on external dataset (Karolinska)  
- Extensive **ablation study** validating the contribution  


# Proposed Method

Traditional MIL models often suffer from "unstable attention distributions" when applied to data from different institutions.

We address this by introducing:

- Attention Regularization
- Entropy term → stabilizes attention  
- Sparsity term → focuses on important patches  

Final Loss Function:
Loss = Classification Loss + λ × Attention Regularization


# Repository Structure

project/
|--- dinov2_features.py # Feature extraction using DINOv2
|--- mil_training.py # Baseline MIL models (CLAM, TransMIL)
|--- domain_regularized_mil.py # Proposed method 
|--- ablation_study.py # Ablation study 
|
|--- data/
| |--- patches/ # Input patches
│ |--- features/ # Extracted features (.npy)
│ |--- splits/ # Train/val/test CSV files
|
|--- checkpoints/ # Saved models
|--- results/ # Predictions and outputs



#Dataset

We use the PANDA (Prostate cANcer graDe Assessment) dataset.

- Train: Radboud  
- Test: Karolinska (cross-institution evaluation)

Dataset link:  
https://www.kaggle.com/c/prostate-cancer-grade-assessment

# Installation

bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

pip install -r requirements.txt


# Usage
1 -  Extract Features
 - python dinov2_features.py

2 - Train Baseline MIL Model
 - python mil_training.py

3 - Train Proposed Method 
 - python domain_regularized_mil.py
4 - Run Ablation Study 
 - python ablation_study.py

# Results - 

| Model                        | Radboud AUC | Karolinska AUC |
| ---------------------------- | ----------- | -------------- |
| CLAM (DINOv2)                | 0.868       | 0.731          |
| TransMIL (DINOv2)            | 0.872       | 0.737          |
| Proposed Method              | 0.872       | 0.811          |
| Ablation (No Regularization) | 0.941       | 0.713          |


The proposed method significantly improves cross-domain performance.

# Ablation Study
The ablation study confirms that:
 - Removing attention regularization leads to performance drop
 - The improvement is directly due to the proposed method

# Requirements
 - Python 3.8+
 - PyTorch
 - NumPy
 - Pandas
 - scikit-learn
 - torchvision
 - tqdm

# Note - 
 - Dataset is not included due to size limitations
 - Please place data inside the /data folder as described

# Acknowledgements
 - DINOv2 by Meta AI
 - PANDA dataset contributors

# Contact
 - Rohit Kumar  
 - Email: kambojrohit42@gmail.com/rohit1043cse.phd24@chitkara.edu.in  
 - GitHub: https://github.com/Rohit12345678910
