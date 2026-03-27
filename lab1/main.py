import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================读取数据=====================
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("Concrete_Data_Yeh.csv")

#特征与标签
X = df.drop("csMPa", axis=1).values
y = df["csMPa"].values

# =====================相关性分析=====================
corr = df.corr()
print("特征与抗压强度相关性：")
print(corr["csMPa"].sort_values(ascending=False))

# =====================划分训练集/测试集8:2=====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================标准化=====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================转为PyTorch张量=====================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

#构建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================构建三层神经网络=====================
class ConcreteNet(nn.Module):
    def __init__(self, input_dim=8):
        super(ConcreteNet, self).__init__()
        #三层网络：输入层→隐藏层1→隐藏层2→输出层
        self.fc1 = nn.Linear(input_dim, 64)   #第一层
        self.fc2 = nn.Linear(64, 32)       #第二层
        self.fc3 = nn.Linear(32, 16)         #第三层
        self.out = nn.Linear(16, 1)          #输出
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x

#初始化模型
model = ConcreteNet(input_dim=8)
print(model)

# =====================损失函数与优化器=====================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================训练模型=====================
epochs = 300
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# =====================测试与评估=====================
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()
    y_true = y_test_tensor.numpy().flatten()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\n=====模型评估=====")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# =====================绘图=====================
plt.figure(figsize=(12, 4))

#损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve")
plt.grid(True)

#真实值 vs 预测值
plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel("True csMPa")
plt.ylabel("Predicted csMPa")
plt.title("True vs Predicted")
plt.grid(True)

plt.tight_layout()
plt.show()