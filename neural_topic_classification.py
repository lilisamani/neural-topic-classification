import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Neural topic classification function using PyTorch
def neural_topic_classification(embeddings_file, epochs, batch_size, output_model):
    data = pd.read_pickle(embeddings_file)

    train_data = data[data["dataset"] == "train"]
    X_train = np.array(train_data["embedding"].tolist(), dtype=np.float32)
    label_names = sorted(train_data["label"].unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y_train = train_data["label"].map(label_to_id).values

    dev_data = data[data["dataset"] == "dev"]
    X_dev = np.array(dev_data["embedding"].tolist(), dtype=np.float32)
    y_dev = dev_data["label"].map(label_to_id).values

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dev_dataset = TensorDataset(
        torch.tensor(X_dev, dtype=torch.float32),
        torch.tensor(y_dev, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # PyTorch Neural Network Model definition
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 16),
        nn.ReLU(),
        nn.Linear(16, len(label_to_id)),
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_correct = 0
            for X_batch, y_batch in dev_loader:
                outputs = model(X_batch)
                dev_correct += (outputs.argmax(dim=1) == y_batch).sum().item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Validation Accuracy: {100 * dev_correct / len(dev_loader.dataset):.2f}%"
        )

    # Save the trained model to the output file
    torch.save(model.state_dict(), output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural topic classification training script.")
    parser.add_argument(
        "embeddings_file",
        type=Path,
        help="Input embeddings file path.",
    )
    parser.add_argument(
        "epochs",
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "output_model",
        type=Path,
        help="Output model file path.",
    )

    args = parser.parse_args()

    neural_topic_classification(
        args.embeddings_file,
        args.epochs,
        args.batch_size,
        args.output_model,
    )
