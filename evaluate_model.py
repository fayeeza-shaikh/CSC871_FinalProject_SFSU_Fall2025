# %%
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample function to simulate predictions (replace with real model later)
def generate_dummy_predictions(n=200):
    y_true = torch.randint(0, 2, (n,)).numpy()
    y_pred = torch.randint(0, 2, (n,)).numpy()
    return y_true, y_pred

# Evaluate and print metrics
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return cm

# Plot the confusion matrix
def plot_confusion_matrix(cm, labels=['NORMAL', 'PNEUMONIA']):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# Main
if __name__ == "__main__":
    # Simulate test (replace this with actual labels + predictions later)
    y_true, y_pred = generate_dummy_predictions(300)

    # Evaluate
    cm = evaluate_model(y_true, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(cm)



