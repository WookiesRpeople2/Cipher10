import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import NuraleNet

# Hyperparameters
dataset_path = "./data/cifar10_train.pt"
model_save_path = "./pytorch/trained.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 1
lr = 0.001
# ------


def train(model, train_dl, num_epochs=num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_batch_loss = 0.0
        for i, (image, label) in enumerate(train_dl):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()  # backward propergation
            optimizer.step()

            running_batch_loss += loss.item()

            if i % 1500 == 1499:
                print("Epoche: {}, Loss: {}".format(
                    epoch, running_batch_loss / 1500))
                running_batch_loss = 0.0


if __name__ == "__main__":
    dataset = torch.load(dataset_path)
    train_dl = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    model = NuraleNet().to(device)
    train(model, train_dl)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_save_path)
