import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import NuraleNet

# Hyperparameters
dataset_path = "./data/cifar10_test.pt"
model_save_path = "./pytorch/trained.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------


def drawplt(model, data):
    def unnormalize(image):
        img = image / 2 + 0.5
        return img

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, _ = next(data)
    images = images.to(device)
    outputs = model(images)
    _, labels = torch.max(outputs, 1)

    for image, label in zip(images, labels):
        image = unnormalize(image.cpu())
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.title(classes[label])
    plt.show()


if __name__ == "__main__":
    model = NuraleNet()
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])

    dataset = torch.load(dataset_path)
    test_dl = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    drawplt(model, iter(test_dl))
