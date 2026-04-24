import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

drive.mount('/content/drive')
data_path=('/content/drive/MyDrive/Vehicles')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

calculated_mean = [0.5078317523002625, 0.5300400257110596, 0.5461992025375366]
calculated_std  = [0.21439097821712494, 0.21168957650661469, 0.22215504944324493]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=calculated_mean, std=calculated_std)
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=calculated_mean, std=calculated_std)
])

train_data = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
val_data   = datasets.ImageFolder(root=os.path.join(data_path, 'dev'),   transform=test_transform)
test_data  = datasets.ImageFolder(root=os.path.join(data_path, 'test'),  transform=test_transform)


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=16, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.flatten_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = self.pool3(x)

        x = x.view(-1, self.flatten_dim)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()


        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_ensemble_with_tta(models, loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images_flipped = torch.flip(images, dims=[3])
            total_outputs = torch.zeros(images.size(0), 3).to(device)

            for model in models:
                model.eval()
                out_original = model(images)
                out_flipped = model(images_flipped)
                total_outputs += out_original + out_flipped

            _, predicted = torch.max(total_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

ensemble_models = []
num_models = 3
for i in range(num_models):
    print(f"Training Model {i+1}/{num_models}...")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    train_model(model, train_loader, val_loader, criterion, optimizer,scheduler, num_epochs=30)

    ensemble_models.append(model)

    acc = evaluate_model(model, test_loader)
    print(f"Model {i+1} Test Acc: {acc:.2f}%")
    print("-" * 30)

final_acc = evaluate_ensemble_with_tta(ensemble_models, test_loader)
print(f"Final acc: {final_acc:.2f}%")

calc_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
temp_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=calc_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)

def get_mean_std(loader):
    mean = 0.
    std = 0.
    total_images = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()

calculated_mean, calculated_std = get_mean_std(temp_loader)
print(f"Calculated Mean: {calculated_mean}")
print(f"Calculated Std:  {calculated_std}")
print("-" * 30)

def visualize_feature_maps(model, test_loader):
    model.eval()

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:4].to(device)

    out1 = model.pool1(F.relu(model.bn1(model.conv1(images))))

    out2 = model.pool2(F.relu(model.bn2(model.conv2(out1))))

    out3 = F.relu(model.bn3(model.conv3(out2)))

    layer_outputs = [out1, out2, out3]
    layer_names = ["Layer 1 (Low-Level Features)", "Layer 2 (Mid-Level Features)", "Layer 3 (High-Level Features)"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    print("Original Images:")
    for i in range(4):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array(calculated_std) + np.array(calculated_mean)
        img = np.clip(img, 0, 1)

        try:
            label_name = train_dataset.classes[labels[i]]
        except:
            label_name = str(labels[i].item())

        axes[i].imshow(img)
        axes[i].set_title(f"{label_name}")
        axes[i].axis('off')
    plt.show()

    for layer_idx, feature_map in enumerate(layer_outputs):
        print(f"\n--- {layer_names[layer_idx]} ---")
        num_filters = 8
        fig, axes = plt.subplots(4, num_filters, figsize=(16, 8))

        for img_idx in range(4):
            for filter_idx in range(num_filters):
                fmap = feature_map[img_idx, filter_idx, :, :].detach().cpu().numpy()

                ax = axes[img_idx, filter_idx]
                ax.imshow(fmap, cmap='viridis')
                ax.axis('off')

                if img_idx == 0:
                    ax.set_title(f"Filter {filter_idx+1}", fontsize=9)
        plt.show()

visualize_feature_maps(model, test_loader)
