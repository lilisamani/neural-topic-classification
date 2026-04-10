import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Evaluate the trained model on test set, using PyTorch
def evaluate_on_test(embeddings_file, trained_model):
    data = pd.read_pickle(embeddings_file)

    test_data = data[data["dataset"] == "test"]
    label_names = sorted(test_data["label"].unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    X_test = np.array(test_data["embedding"].tolist(), dtype=np.float32)
    y_test = test_data["label"].map(label_to_id).values

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = nn.Sequential(
        nn.Linear(X_test.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 32),
        nn.ReLU(),
        nn.Linear(32, len(label_to_id)),
    )
    model.load_state_dict(torch.load(trained_model))
    model.eval()

    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total if total else 0.0
    print(f"Test Accuracy: {100 * accuracy:.2f}%")

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=int)
    num_labels = len(label_names)
    confusion = np.zeros((num_labels, num_labels), dtype=int)
    for true_id, pred_id in zip(y_true, y_pred):
        confusion[true_id, pred_id] += 1

    header = "true\\pred".ljust(20) + "".join(name.rjust(20) for name in label_names)
    print("Confusion Matrix:")
    print(header)
    for row_idx, row in enumerate(confusion):
        row_label = label_names[row_idx].ljust(20)
        row_vals = "".join(str(val).rjust(20) for val in row)
        print(f"{row_label}{row_vals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural topic classification evaluation script.")
    parser.add_argument(
        "embeddings_file",
        type=Path,
        help="Input embeddings file path.",
    )
    parser.add_argument(
        "trained_model_file",
        type=Path,
        help="Path to the trained model file.",
    )

    args = parser.parse_args()

    evaluate_on_test(
        args.embeddings_file,
        args.trained_model_file,
    )
