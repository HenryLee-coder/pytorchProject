# encoding:utf-8

# 1、导入包
from pathlib import Path
import torch
import torchvision
import requests
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import gzip
import numpy as np

# 2、数据处理
# 构建基本地址
DATA_PATH = Path("./jupyter_file/分类与回归/data")
PATH = DATA_PATH/"mnist"
PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist"
FILENAME = "mnist.pkl.gz"

if not (PATH/FILENAME).exists():
    content = requests.get(URL+ FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
# 使用 pickle.load() 函数将其反序列化为 Python 对象
# with 确保在结束操作后自动释放资源。不需要再手动调用 f.close() 函数来关闭文件了
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train),(x_valid, y_valid),_) = pickle.load(f, encoding="latin_1")

# 使用torch将数据转化为Tensor
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

print(x_train.shape)

bs = 64
# 关键一步使用DATASET与DATALOADER制作数据集
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True)




# 3、模型搭建
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


model = Mnist_NN()
# 指定损失
cost = torch.nn.CrossEntropyLoss()
# 使用Adam优化器
opt_model = torch.optim.Adam(model.parameters(), lr=0.0001)


# 4、模型训练
losses = []
for step in range(1000):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for xb, yb in train_dl:
        loss = cost(model(xb), yb)
        opt_model.zero_grad()
        loss.backward(retain_graph=True)
        opt_model.step()
        batch_loss.append(loss.data.numpy())

    if step % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(step, np.mean(batch_loss))




# 5、模型验证





if __name__ == '__main__':
    pass
