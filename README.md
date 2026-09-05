# Intelligent Iris Species Classification (Perceptron & Logistic Regression from Scratch)

A machine learning project that classifies iris flowers into three species — **Setosa, Versicolor, and Virginica** — by implementing and comparing three classification strategies **from scratch** (no scikit-learn classifiers), all extended to multi-class classification using a **One-vs-Rest (OvR)** strategy.

## Overview

This project compares three learning strategies built on the same core idea — finding a decision boundary that separates the flower classes:

1. **Perceptron (Step Function)** — the simplest hard-threshold decision rule
2. **Perceptron (Sigmoid Function)** — a smoother, probability-based version of the perceptron
3. **Logistic Regression** — trained with full batch gradient descent

Since the Iris dataset has 3 classes and all three base models are binary classifiers, each strategy is extended using **One-vs-Rest**: three binary classifiers are trained per strategy (e.g. *Setosa vs Rest*, *Versicolor vs Rest*, *Virginica vs Rest*), and the final prediction is the class with the highest score (argmax).

## Dataset

The classic **Iris dataset** (via `sklearn.datasets`) — 150 samples, 50 per species, with 4 numeric features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Pipeline

1. **Data Preprocessing** — load the dataset
2. **Feature Scaling** — standardize features using `StandardScaler`
3. **Train/Test Split** — 80/20 stratified split (120 train / 30 test samples)
4. **Model Training** — train all three strategies using One-vs-Rest
5. **Evaluation** — compute and compare accuracy across the three models
6. **Visualization** — bar chart comparing model accuracy
7. **Inference** — predict the species of a new, unseen flower sample

## Results

| Model                    | Accuracy |
|---------------------------|----------|
| Perceptron (Step)         | ~70%     |
| Perceptron (Sigmoid)      | ~73%     |
| Logistic Regression       | ~93%     |

Logistic Regression consistently outperforms both perceptron variants, since it updates weights using full batch gradient descent over the entire training set rather than one random sample at a time — leading to smoother, more stable convergence.

> **Note:** The Step and Sigmoid Perceptrons use random sample selection during training without a fixed seed, so their accuracy can vary slightly between runs. Logistic Regression's accuracy is deterministic.

## Tech Stack

- Python 3
- NumPy
- Pandas
- scikit-learn (dataset loading & preprocessing utilities only — no built-in classifiers used)
- Matplotlib
- Tabulate

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/MuhammadBilal00011/iris-classification-ovr.git
cd iris-classification-ovr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python main.py
```

## Project Structure

```
.
├── main.py             # Full pipeline: preprocessing, training, evaluation, visualization
├── requirements.txt    # Python dependencies
├── Project_Report.docx # Detailed project report
└── README.md
```

## Key Concepts Demonstrated

- Feature scaling and stratified train/test splitting
- Binary classifiers built from scratch (Perceptron, Sigmoid, Logistic Regression)
- Gradient descent optimization
- Multi-class classification via One-vs-Rest decomposition
- Numerical stability techniques (input clipping in the sigmoid function)

## Author

**Muhammad Bilal**
[GitHub](https://github.com/MuhammadBilal00011)
