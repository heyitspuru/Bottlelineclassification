The repo containes the full Implemented code ipynb file with the original dataset from cocacola fault detector, and a sample folder of cleaned/cropped images dataset that is used in the process. while executing this , We had to make sure we cleaned the dataset manually. If you require the full cleaned dataset drop a dm @pururajdhama2004@gmail.com

# Bottle Fill-Level Classification Using CNN and MobileNetV2

This repository presents a deep learning-based project for classifying bottle images into three categories based on their fill level: **Underfilled**, **Overfilled**, and **Normal filled**. The implementation includes a custom CNN model and a transfer learning approach using **MobileNetV2**. Additionally, an unsupervised learning attempt using K-Means and DBSCAN was explored and discussed.

## 🧠 Project Overview

- **Objective:** Automate the classification of bottle fill levels from images to improve industrial inspection processes.
- **Techniques Used:**
  - Custom CNN model with hyperparameter tuning
  - Transfer Learning with MobileNetV2
  - Unsupervised clustering with K-Means and DBSCAN (exploratory)

## 🏗️ Project Structure

├── dataset/
│   ├── images/
│   │   ├── underfilled/
│   │   ├── overfilled/
│   │   └── normalfilled/
│   └── augmented_images/
├── notebooks/
│   ├── cnn_model.ipynb
│   ├── mobilenetv2_model.ipynb
│   └── unsupervised_learning.ipynb
├── models/
│   ├── cnn_best_model.h5
│   └── mobilenetv2_model.h5
├── results/
│   ├── classification_reports/
│   └── confusion_matrices/
├── README.md
└── requirements.txt

📊 Results
| Model                 | Accuracy | Precision | Recall | F1-Score |
| --------------------- | -------- | --------- | ------ | -------- |
| Custom CNN (baseline) | 71%      | 0.72      | 0.71   | 0.71     |
| Custom CNN (tuned)    | 98%      | 0.98      | 0.98   | 0.98     |
| MobileNetV2           | 98%      | 0.98      | 0.98   | 0.98     |

MobileNetV2 outperformed the baseline model in all metrics and demonstrated superior generalization and class balance handling.

🛠️ Tech Stack
Language: Python 3.10+

Environment: Google Colab

Deep Learning: TensorFlow 2.13.0, Keras

Hyperparameter Tuning: Keras Tuner 1.3.5

Preprocessing: OpenCV 4.8.0.76, Pillow 10.0.0

Data Handling: NumPy, Pandas 2.0.3

Evaluation: Scikit-learn 1.3.0, Matplotlib 3.7.1

🧪 How to Run
Clone the repository:
Copy
Edit
git clone https://github.com/yourusername/bottle-fill-level-classification.git
cd bottle-fill-level-classification

Install the required libraries:
Copy
Edit
pip install -r requirements.txt
Run the notebooks:

notebooks/cnn_model.ipynb

notebooks/mobilenetv2_model.ipynb

notebooks/unsupervised_learning.ipynb

🤝 Contributions
Feel free to fork this repo, make improvements, and submit a pull request!
