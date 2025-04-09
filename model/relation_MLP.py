import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Relation_MLP(nn.Module):
    def __init__(self, input_size=1024, num_classes=2):
        super(Relation_MLP, self).__init__()
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
    
    
def relation_embedding(feature_list):
    res_mrt = np.zeros((len(feature_list),len(feature_list)))
    model = Relation_MLP().to(device)
    model.load_state_dict(torch.load(r"D:\frac\zzhNet\model_epoch_100.pth"))  # 载入已保存模型
    model.eval()
    for i in len(feature_list):
        for j in range(i-1):
            result = model(torch.cat([feature_list[i], feature_list[j]], dim=0))  # shape: (2, 1024)
            res_mrt[i][j] = result
            res_mrt[j][i] = result
    return res_mrt

# 生成示例数据
data = np.load(r"D:\frac\zzhNet\dataset2.npz")
x_t = data["X_train"]
y_t = data["y_train"]
x_t = x_t.squeeze(axis=2)

# 转换为 PyTorch 张量
X_tensor = torch.from_numpy(x_t)
y_tensor = torch.from_numpy(y_t)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Relation_MLP().to(device)
criterion = nn.NLLLoss()  # 使用 NLLLoss 适配 LogSoftmax 输出
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 余弦退火学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 训练
num_epochs = 200  # 设为 200 轮，以体现保存间隔
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 更新学习率
    scheduler.step()

    # 打印信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 每 100 轮保存模型
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}.")







