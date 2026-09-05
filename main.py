"""01. Import Libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

"""02. Load and Print Data"""
iris = datasets.load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = iris.target
print("\n", "Iris Flowers Data")
print("=" * 80)
print(X.head())
print("\n","Classes (target values):","0 = Setosa,","1 = Versicolor,","2 = Virginica")

"""03. Split Data"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

"""04. Feature Scaling"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""05. Step Function"""
def step(z):
    return 1 if z >= 0 else 0

"""06. Sigmoid Function"""
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

"""07. Perceptron(step) Function Training"""
def train_perceptron(X, y, lr=0.1, epochs=1000):
    X_b = np.c_[np.ones(len(y)), X]
    weights = np.ones(X_b.shape[1])

    for _ in range(epochs):
        j = np.random.randint(0, len(y))
        y_hat = step(np.dot(X_b[j], weights))
        weights += lr * (y[j] - y_hat) * X_b[j]
    return weights

def predict_perceptron(X, weights):
    X_b = np.c_[np.ones(X.shape[0]), X]
    return np.array([step(np.dot(x, weights))for x in X_b])

"""08. Sigmoid Function Training"""
def train_sigmoid_model(X, y, lr=0.1, epochs=1000):
    X_b = np.c_[np.ones(len(y)), X]
    weights = np.ones(X_b.shape[1])

    for _ in range(epochs):
        j = np.random.randint(0, len(y))
        z = np.dot(X_b[j], weights)
        y_hat = sigmoid(z)
        error = (y[j] - y_hat) * y_hat * (1 - y_hat)
        weights += lr * error * X_b[j]
    return weights

def predict_sigmoid(X, weights):
    X_b = np.c_[np.ones(len(X)), X]
    return np.array([
        1 if sigmoid(np.dot(x, weights)) >= 0.5 else 0 for x in X_b])

"""09. Logistic Regression Training"""
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0.0

    for _ in range(epochs):
        z = np.dot(X, weights) + bias
        y_hat = sigmoid(z)
        error = y - y_hat
        weights += lr * np.dot(X.T, error)
        bias += lr * np.sum(error)
    return weights, bias

def predict_logistic(X, weights, bias):
    probabilities = sigmoid(np.dot(X, weights) + bias)
    return (probabilities >= 0.5).astype(int)

"""10. One vs Rest Function (train three binary classifiers using One vs Rest.)"""
def train_ovr(model_name, X_train, y_train):
    models = {}
    for class_id in [0, 1, 2]:
        y_binary = (y_train == class_id).astype(int)

        if model_name == "step":
            weights = train_perceptron(X_train,y_binary)
            models[class_id] = weights

        elif model_name == "sigmoid":
            weights = train_sigmoid_model(X_train,y_binary)
            models[class_id] = weights

        elif model_name == "logistic":
            weights, bias = train_logistic_regression(X_train,y_binary)
            models[class_id] = (weights, bias)
    return models

"""11. One vs Rest Function Prediction"""
def predict_ovr(model_name, models, X_test):
    all_scores = []

    for class_id in [0, 1, 2]:

        if model_name == "step":
            weights = models[class_id]
            X_b = np.c_[np.ones(len(X_test)), X_test]
            scores = np.array([step(np.dot(x, weights)) for x in X_b])

        elif model_name == "sigmoid":
            weights = models[class_id]
            X_b = np.c_[np.ones(len(X_test)), X_test]
            scores = np.array([sigmoid(np.dot(x, weights)) for x in X_b])

        elif model_name == "logistic":
            weights, bias = models[class_id]
            scores = sigmoid(np.dot(X_test, weights) + bias)

        all_scores.append(scores)

    all_scores = np.array(all_scores).T
    predictions = np.argmax(all_scores, axis=1)
    return predictions, all_scores

"""12. Train all Three Models"""
step_models = train_ovr("step",X_train,y_train)

sigmoid_models = train_ovr("sigmoid",X_train,y_train)

logistic_models = train_ovr("logistic",X_train,y_train)

"""13. Predictions"""
step_pred, step_scores = predict_ovr("step",step_models,X_test)

sigmoid_pred, sigmoid_scores = predict_ovr("sigmoid",sigmoid_models,X_test)

logistic_pred, logistic_scores = predict_ovr("logistic",logistic_models,X_test)

"""14. Accuracy"""
step_accuracy = np.mean(step_pred == y_test)
sigmoid_accuracy = np.mean(sigmoid_pred == y_test)
logistic_accuracy = np.mean(logistic_pred == y_test)

print("\n","RESULTS")
print("=" * 60)
print("Perceptron (Step Function) Accuracy:",step_accuracy)
print("Perceptron (Sigmoid) Accuracy:",sigmoid_accuracy)
print("Logistic Regression Accuracy:",logistic_accuracy)

"""15. Comparision Table"""
results = pd.DataFrame({
    "Model": ["Perceptron (Step)","Perceptron (Sigmoid)","Logistic Regression"],
    "Accuracy": [step_accuracy,sigmoid_accuracy,logistic_accuracy]})
results["Accuracy"] = results["Accuracy"] * 100

print("\n","Model performance Comparision")
print("=" * 50)
print(tabulate(results,headers="keys",tablefmt="fancy_grid",showindex=False,floatfmt=".2f"))

"""16. Accuracy Graph"""
plt.figure(figsize=(8, 5))
colors = ["#4C78A8", "#F58518", "#54A24B"]
bars = plt.bar(results["Model"], results["Accuracy"], color=colors,width=0.6,edgecolor="black",linewidth=1.2)
plt.bar_label(bars, fmt='%.2f%%', padding=3, fontweight='bold')
plt.ylim(0, 100)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Iris Classification - OvR Model Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

"""17. Predict one new flower"""
"""Example input:
[sepal length, sepal width, petal length, petal width]
The same scaler used during training must be used here."""
"""new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])"""
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
new_flower_scaled = scaler.transform(new_flower)

step_new, _ = predict_ovr("step",step_models,new_flower_scaled)
sigmoid_new, _ = predict_ovr("sigmoid",sigmoid_models,new_flower_scaled)
logistic_new, _ = predict_ovr("logistic",logistic_models,new_flower_scaled)

print("\n","New Flower Prediction")
print("=" * 50)
print("Perceptron (Step):",iris.target_names[step_new[0]])
print("Perceptron (Sigmoid):",iris.target_names[sigmoid_new[0]])
print("Logistic Regression:",iris.target_names[logistic_new[0]])