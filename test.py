import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size=1024, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 适用于分类任务

    def forward(self, x):
        x = x.view(x.shape[0], 2, 1024)  # 保持 (batch_size, 2, 1024)
        x = x.mean(dim=1)  # 在维度 1 进行均值池化，变成 (batch_size, 1024)
        
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.logsoftmax(x)  # 适用于 NLLLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MLP().to(device)
model.load_state_dict(torch.load(r"D:\frac\zzhNet\model_epoch_100.pth"))  # 载入已保存模型
model.eval()  # 设置为评估模式

# 生成随机数据作为 X（假设形状为 (1000, 2, 1024)）
X = np.random.rand(1000, 2, 1024).astype(np.float32)  # (1000, 2, 1024)

# 选择 5 个测试样本
indices = np.random.choice(1000, 100, replace=False)  # 随机选 5 个索引
test_samples = torch.tensor(X[indices]).to(device)  # 转换为 PyTorch Tensor

# 前向传播
with torch.no_grad():
    predictions = model(test_samples)
    predicted_classes = torch.argmax(predictions, dim=1)  # 取最大值的索引（类别）

# 输出结果
for i, idx in enumerate(indices):
    print(f"Sample {idx}: Predicted Class = {predicted_classes[i].item()}")