# Robust Fake Profile Detection Using Deep Neural Networks with Adversarial Training

This repository contains a project aimed at detecting fake Instagram profiles using Deep Neural Networks (DNNs). To enhance resilience against adversarial attacks, the model leverages adversarial training, resulting in improved robustness compared to a standard DNN.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Adversarial Attacks](#adversarial-attacks)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview
Fake social media profiles are often used for misinformation, identity theft, and other malicious activities. This project focuses on detecting such profiles on Instagram by training two models:

1. **Standard DNN** (trained solely on clean data).  
2. **Robust DNN** (trained with both clean and adversarial examples).

The **Robust DNN** offers improved resilience when tested against popular adversarial attacks such as FGSM, PGD, and CW.

---

## Key Features
- **Data Preprocessing**: Automatic handling of missing values, feature standardization, and class imbalance mitigation via SMOTE.  
- **Custom DNN Architecture**: A multi-layer feedforward network (PyTorch) with dropout and ReLU activations.  
- **Adversarial Training**: Incorporation of PGD-based adversarial examples during training to bolster model robustness.  
- **Attack Evaluation**: Resilience tested using FGSM, PGD, and Carlini–Wagner attacks.  
- **Performance Metrics**: Accuracy, precision, recall, and F1-score (per class and macro/weighted).

---

## Project Structure

.
├── RobustFakeProfileDetection.ipynb   # Jupyter Notebook implementing the approach
├── instagram_dataset.csv              # Dataset (236 profiles: 28 real, 208 fake)
├── README.md                          # This README file
├── requirements.txt                   # Python dependencies
└── …

- **RobustFakeProfileDetection.ipynb**: Demonstrates data preprocessing, model training, adversarial attack generation, and evaluation.  
- **instagram_dataset.csv**: Original dataset used in this project.  
- **requirements.txt**: A list of Python packages needed to run this code.

---

## Getting Started

### Prerequisites
- Python 3.7 (or later)
- [PyTorch](https://pytorch.org/) (v1.0 or later)
- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- (Optional) Libraries for adversarial attacks (e.g., [Foolbox](https://github.com/bethgelab/foolbox) or custom code)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/RobustFakeProfileDetection.git
   cd RobustFakeProfileDetection

	2.	Install dependencies:

pip install -r requirements.txt


	3.	Open the notebook (if using Jupyter):

jupyter notebook

Then navigate to RobustFakeProfileDetection.ipynb.

⸻

Usage
	1.	Data Preparation
	•	Ensure instagram_dataset.csv is in the same directory (or update the path in the notebook).
	•	Run the Data Preprocessing cells to split the dataset, handle SMOTE, and normalize features.
	2.	Model Training
	•	Within RobustFakeProfileDetection.ipynb, select either:
	•	Standard DNN training (clean data only).
	•	Robust DNN training (clean + adversarial examples).
	•	Hyperparameters (e.g., learning rate, batch size, epochs, epsilon) can be adjusted as needed.
	3.	Adversarial Attack Evaluation
	•	After training, generate adversarial samples (FGSM, PGD, CW) and measure accuracy/precision/recall/F1.
	•	Compare the Standard DNN vs. Robust DNN performance under each attack.
	4.	Command-Line Execution (if converted to a Python script):

python robust_fake_profile_detection.py

Adjust script arguments for custom paths or parameters.

⸻

Methodology

Dataset
	•	Source: instagram_dataset.csv with 236 labeled Instagram profiles (real vs. fake).
	•	Features:
	•	Numerical: Follower count, following count, username length, etc.
	•	Boolean: Whether the account is private, business, recently joined, etc.
	•	Preprocessing:
	•	Dropped irrelevant columns like has_channel and has_guides.
	•	Applied SMOTE to handle class imbalance (28 real vs. 208 fake).
	•	Used StandardScaler to normalize feature distributions.

Model Architecture
	•	Layers:
	•	Input: 64 neurons (post feature transformation).
	•	Hidden: Two dense layers (32 neurons each) with ReLU activations and dropout (0.2, 0.3).
	•	Output: 2 neurons with softmax for binary classification.
	•	Training:
	•	Optimizer: Adam (learning rate = 0.001).
	•	Loss: Weighted cross-entropy (to offset imbalance).
	•	Epochs: 200.

Adversarial Attacks
	•	FGSM: Fast Gradient Sign Method.
	•	PGD: Projected Gradient Descent.
	•	CW: Carlini–Wagner.

The Robust DNN is trained with a blend of clean and PGD adversarial examples each epoch, yielding higher defense against all tested attacks.

⸻

Results

Model	Scenario	Accuracy	Real (Prec / Rec)	Fake (Prec / Rec)
Standard DNN	Clean	0.92	(0.65 / 0.79)	(0.97 / 0.94)
	FGSM	0.88	(0.00 / 0.00)	(0.88 / 1.00)
	PGD	0.88	(0.00 / 0.00)	(0.88 / 1.00)
	CW	0.83	(0.00 / 0.00)	(0.88 / 0.94)
Robust DNN	Clean	0.91	(0.57 / 0.82)	(0.97 / 0.92)
	FGSM	0.88	(0.50 / 0.82)	(0.97 / 0.89)
	PGD	0.88	(0.50 / 0.82)	(0.97 / 0.89)
	CW	0.88	(0.49 / 0.82)	(0.97 / 0.88)

Observations
	•	Standard DNN: Performs well on clean data but drops significantly under adversarial conditions (failing to detect real profiles).
	•	Robust DNN: Slightly lower clean-data performance but maintains ~88% accuracy under all attacks and consistently detects both classes.

⸻

Future Work
	•	Larger Dataset: Improve generalization and reduce overfitting with more data.
	•	Multi-Attack Training: Include FGSM and CW examples in the training process for broader robustness.
	•	Advanced Architectures: Explore graph neural networks or ensembles to further bolster performance.
	•	Refined Oversampling: Use cost-sensitive approaches or advanced data augmentation for real-profile minority class.

⸻

Contributing

Contributions are welcome! Feel free to:
	•	Fork this repository
	•	Create a new branch
	•	Submit a pull request

Please open an issue for significant design changes to discuss the proposal first.

⸻

License

This project is licensed under the MIT License. See the LICENSE file for details.

⸻

References
	1.	D. Guna Sherar et al., “Fake Profile Detection Using Deep Learning Algorithm,” IRJET, 2024.
	2.	Chongyang Zhao et al., “Adversarial Example Detection for Deep Neural Networks: A Review,” IEEE DSC, 2023.
	3.	Eben Charles & Ponnarasan Krishnan, “Adversarial Attacks in Deep Learning: Analyzing Vulnerabilities and Designing Robust Defense Mechanisms,” Feb 2024.

