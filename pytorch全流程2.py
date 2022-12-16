# encoding:utf-8

# 1.导包
import numpy as np
import torch
import torchvision
import torch.nn as nn
#使用现有数据集
from torchvision import datasets, transforms
#自己构建数据集
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# 2、数据处理
# 超参数
PATH_DATA = Path("./jupyter_file/卷积网络参数/data")
#input_size = 28
#num_clases = 10
batch_size = 64

# 训练集及batch数据
train_dataset = datasets.MNIST(root=PATH_DATA,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# 测试集及batch数据
test_dataset = datasets.MNIST(root=PATH_DATA,
                              train=False,
                              transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)

# 3、模型构建 使用sequential


class CNN(nn.Module):
    def __init__(self):
        # 初始化父类方法，以调用父类属性
        super(CNN, self).__init__()
        # 第一组卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )

        self.conv3 = nn.Sequential(  # 下一个套餐的输入 (32, 7, 7)
            nn.Conv2d(32, 64, 3, 1, 1),  # 输出 (64, 7, 7)
            nn.ReLU(),  # relu层
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.MaxPool2d(2),                # 输出 (32, 7, 7)
        )

        self.conv4 = nn.Sequential(  # 下一个套餐的输入 (32, 7, 7)
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # 输出 (64, 7, 7)
        )

        self.out = nn.Linear(64 * 7 * 7, 10)  # 全连接层得到的结果

    # 前向推理
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x
# 定义精度
def accuracy(predict, labels):
    pred = torch.max(predict.data, 1)[1]
    right = pred.eq(labels.data.view_as(pred)).sum()
    return right, len(labels)


# 4、模型训练

def train(num_epoches):
    module = CNN()  # 实例化模型
    loss_CEL = nn.CrossEntropyLoss()  # 损失函数
    opt_param = torch.optim.Adam(module.parameters(), lr=0.001)  # 优化器
    # 循环训练,先定义循环总次数，在每个循环把总量分成批次训练
    for epoch in range(num_epoches):
        train_resulte = []  # 保存本次的epoch结果
        # 按批次开始,enumerate遍历一个可迭代对象返回；两个信息，第一个是索引，第二个是数据本身
        for batch_idx, (soure, target) in enumerate(train_loader):
            module.train()
            output = module(soure)  # 输出预测值
            loss = loss_CEL(output, target)   # 计算loss
            opt_param.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            opt_param.step()  # 更新梯度
            right = accuracy(output, target)  # 统计accuracy
            train_resulte.append(right)  # 保存正确结果
            # 走一定批次验证一次，正常应该独立开
            if batch_idx % 100 == 0:
                module.eval()
                val_rights = []

                for (data, target) in test_loader:
                    output = module(data)
                    right = accuracy(output, target)
                    val_rights.append(right)
                # 准确率计算
                train_r = (sum([tup[0] for tup in train_resulte]), sum([tup[1] for tup in train_resulte]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data,
                    100. * train_r[0].numpy() / train_r[1],
                    100. * val_r[0].numpy() / val_r[1]))


if __name__ == '__main__':
    train(3)
