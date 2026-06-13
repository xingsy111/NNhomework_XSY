"""
训练作业2的SVHN CNN模型，生成FP32权重文件供作业4使用。
基于 homework_2/main.py 的模型结构。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# ==================== 数据集类 ====================
class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        print(f"加载数据集: {mat_file}")
        data = loadmat(mat_file)
        self.images = data['X'].transpose(3, 2, 0, 1)  # (N, C, H, W)
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0  # SVHN中10代表数字0
        self.transform = transform
        print(f"数据集大小: {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image_pil = Image.fromarray(np.transpose(image, (1, 2, 0)))
            image = self.transform(image_pil)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image, label

# ==================== CNN模型 (严格匹配作业2) ====================
class SVHNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHNNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== 训练 ====================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100. * correct / total, running_loss / len(train_loader)

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total, running_loss / len(test_loader)

def main():
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_dir = os.path.join(os.path.dirname(__file__), 'homework_2')
    train_dataset = SVHNDataset(os.path.join(data_dir, 'train_32x32.mat'), transform=transform)
    test_dataset = SVHNDataset(os.path.join(data_dir, 'test_32x32.mat'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = SVHNNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 20  # CPU训练，适当减少epoch
    print(f"\n开始训练，总轮数: {num_epochs}\n")

    best_acc = 0
    for epoch in range(num_epochs):
        train_acc, train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_acc, test_loss = test_model(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'svhn_cnn_fp32.pth')
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] *BEST* '
                  f'Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Loss: {test_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] '
                  f'Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Loss: {test_loss:.4f}')

    print(f"\n训练完成，最佳测试准确率: {best_acc:.2f}%")
    print(f"模型权重已保存为: svhn_cnn_fp32.pth")

if __name__ == "__main__":
    main()
