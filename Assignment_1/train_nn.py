from models import MLP, VGG, ResNet, VGGDropout, VGGWeightDecay, BasicBlock
import argparse
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument('--model', type=str, choices=['mlp', 'vgg', 'resnet', 'vgg_dropout', 'vgg_weight_decay'], default='mlp',
                        help='NN model to use (default: mlp)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'adagrad'], default='sgd',
                        help='optimizer to use (default: sgd)')
    parser.add_argument('--transform', type=str, choices=['default', 'random_filp', 'random_crop', 'normalize'], default='default',
                        help='data augumentation to use (default: default)')
    args = parser.parse_args()
    writer = SummaryWriter()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    transform_random_filp = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_random_crop = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(90),
        transforms.Resize((256, 256)),
    ])
    transform_normalize = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    if args.transform == 'random_filp':
        transform = transform_random_filp
    elif args.transform == 'random_crop':
        transform = transform_random_crop
    elif args.transform == 'normalize':
        transform = transform_normalize
    training_dataset = torchvision.datasets.ImageFolder(
        './Data/seg_train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(
        './Data/seg_test', transform=transform)
    training_dataloader = DataLoader(training_dataset, batch_size=32,
                                     shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = MLP(3*256*256, 64*64, 6)
    if args.model == 'vgg':
        model = VGG(6)
    elif args.model == 'vgg_dropout':
        model = VGGDropout(6)
    elif args.model == 'vgg_weight_decay':
        model = VGGWeightDecay(6)
    elif args.model == 'resnet':
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=6)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=0.005)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(), lr=0.005)

    # Training loop
    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(training_dataloader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(training_dataloader)}], Loss: {loss.item():.4f}")
            global_step = epoch * len(training_dataloader) + batch_idx

        model.eval()
        with torch.no_grad():
            # Evaluation on the test set
            correct = 0
            total = 0
            for data, targets in test_dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy on the test set: {accuracy:.2f}%")


if __name__ == '__main__':
    train()
