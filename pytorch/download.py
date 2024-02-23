import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transform, download=True)

torch.save(train_dataset, './data/cifar10_train.pt')
torch.save(test_dataset, './data/cifar10_test.pt')
