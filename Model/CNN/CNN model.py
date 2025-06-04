import tensorflow as tf
# import tf_keras as keras
import keras as keras

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.utils import to_categorical
from keras.layers import Conv1D, Dense, Flatten

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os

class ApplianceDataset(Dataset):
    def __init__(self, root_dir, window_size=100, step_size=50):
        self.samples = []
        self.labels = []
        self.label_map = {}
        label_id = 0

        for appliance_dir in Path(root_dir).iterdir():
            if not appliance_dir.is_dir():
                continue
            label = appliance_dir.name
            if label not in self.label_map:
                self.label_map[label] = label_id
                label_id += 1

            for file in appliance_dir.glob("*.csv"):
                df = pd.read_csv(file)
                if 'Power' not in df.columns:
                    continue

                power = df["Power"].values.astype(np.float32)

                for i in range(0, len(power) - window_size, step_size):
                    segment = power[i:i+window_size]
                    self.samples.append(segment)
                    self.labels.append(self.label_map[label])

        self.samples = torch.tensor(self.samples).unsqueeze(1)  # shape: [N, 1, window_size]
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((input_length // 4) * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train(model, dataloader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()

        acc = 100.0 * correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

def main():
    root_dir = "data"  # your folder with class subfolders
    window_size = 100
    dataset = ApplianceDataset(root_dir, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = len(dataset.label_map)
    model = CNN1D(input_length=window_size, num_classes=num_classes)

    train(model, dataloader, epochs=20, lr=0.001)

if __name__ == "__main__":
    main()

