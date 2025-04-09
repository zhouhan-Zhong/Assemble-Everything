from scipy.spatial.distance import cdist
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def compute_weighted_matrix(adj_matrix):
    n = adj_matrix.shape[0]

    # 计算不同间隔的邻接情况
    A1 = adj_matrix
    A2 = np.linalg.matrix_power(adj_matrix, 2)
    # A3 = np.linalg.matrix_power(adj_matrix, 3)

    # 初始化权重矩阵
    W = np.zeros((n, n))

    # 赋值权重
    W[A1 > 0] = 1      # 间隔 1
    W[(A2 > 0) & (A1 == 0)] = 0.5  # 间隔 2（不包括直接相邻的）
    # W[(A3 > 0) & (A1 == 0) & (A2 == 0)] = 0.25  # 间隔 3（不包括前两者）

    return W
    
class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = Transform()


    def forward(self, x):
        return self.transform(x)
    


def fd_files(directory):
    obj_files = []
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.obj'):
                # 获取文件的完整路径或只获取文件名
                obj_files.append(os.path.join(root, file))
    return obj_files


def load_obj_to_set(file_path):
    vertices = []

    # 打开 .obj 文件
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] == 'v':  # 只处理顶点行
                x, y, z = map(float, parts[1:4])  # 读取坐标
                vertices.append([x, y, z])

    # 转换为 NumPy 数组
    return set(map(tuple,vertices))

def load_obj_as_tensor(file_path):
    vertices = []

    with open(file_path, 'r') as f:
        for line in f:
            # 只解析顶点信息，OBJ 顶点行以 "v" 开头
            if line.startswith('v '):
                parts = line.strip().split()
                # 解析 x, y, z 坐标并转换为 float
                vertex = list(map(float, parts[1:4]))  # 仅取前三个数值
                vertices.append(vertex)

    # 转换为 PyTorch Tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    return vertices_tensor



def are_components_adjacent(comp1, comp2):
    return not comp1.isdisjoint(comp2)

def generate_martrix(num):
    return np.zeros((num,num))



def datamake(dirr):
    dr_list = fd_files(dirr)  # r'D:\frac\fractured_1'
    obj_list = []
    obj_tf = [] 
    mrt = generate_martrix(len(dr_list))

    for i in dr_list:
        obj = load_obj_to_set(i)
        obj_list.append(obj)
        obj_tf.append(load_obj_as_tensor(i))
    # print(obj_list[0])
    for i in range(len(dr_list)):
        for j in range(i-1):
            if are_components_adjacent(obj_list[i],obj_list[j]):
                mrt[i][j] = 1
                mrt[j][i] = 1

            else:
                mrt[i][j] = 0
                mrt[j][i] = 0
    w_mrt = compute_weighted_matrix(mrt)



    pointnet = PointNet()
    checkpoint = torch.load('/kaggle/input/pointnet/pointnet.pth',map_location=torch.device('cuda') )
    pointnet.load_state_dict(checkpoint)
    pointnet = pointnet.to("cuda")
    pointnet.eval()
    fea_list = []

    for i in obj_tf:
        # print(i.transpose(0,1))
        # print(i.shape)
        i = i.transpose(0,1)
        i = i.unsqueeze(0)
        i = i.to("cuda")
        output, a,b = pointnet(i)
        fea_list.append(output.detach().cpu().numpy())
        del i

    del pointnet  # 删除模型变量
    torch.cuda.empty_cache()  # 清理未释放的缓存
    torch.cuda.synchronize()  # 确保所有 GPU 操作完成（可选）

    data = []
    label = []
    for i in range(len(dr_list)):
        for j in range(i-1):
            tp = [fea_list[i],fea_list[j]]
            data.append(tp)
            tp = [fea_list[j],fea_list[i]]
            data.append(tp)

            label.append(w_mrt[i][j])
            label.append(w_mrt[j][i])

    return data, label

def find_fragment_dirs(root_dir):
    fragment_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'fragment' in dirnames:  # 检查当前目录是否有名为 fragment 的子目录
            fragment_dirs.append(os.path.join(dirpath, 'fragment'))
    return fragment_dirs

def main():
    # 示例
    root_directory = "/kaggle/input/data315/data315"  # 替换为你的目录路径
    fragment_dirs = find_fragment_dirs(root_directory)
    all_dir = []
    for i in fragment_dirs:
        for d in os.listdir(i):
            if os.path.isdir(os.path.join(i, d)):
                all_dir.append(os.path.join(i, d))

    print(len(all_dir))
    print(all_dir[1])
    all_data = []
    all_label = []
    for i in all_dir:
        tp_data, tp_label  = datamake(i)  # 
        all_data.extend(tp_data)
        all_label.extend(tp_label)

    X_train = np.array(all_data)  
    y_train = np.array(all_label)  

    print("X_train shape:", X_train.shape)  
    print("y_train shape:", y_train.shape)  
    np.savez("dataset2.npz", X_train=X_train, y_train=y_train) #

main()