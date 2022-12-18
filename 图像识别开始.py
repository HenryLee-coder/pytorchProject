# encoding:utf-8
import copy
# 1、导入包
import os
import time
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

# 2、处理数据
data_dir = './jupyter_file/图像识别与训练/flower_data/'  # 寻找数据根目录
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
# 超参数
input_size = 64
batch_sizes = 128
model_name = 'resnet18'  # 选用resent模型
numclasses = 128  # 种类
my_wight = 'best.pt'  # 权重文件
# 数据形变
data_trainsforms = {
    'train':
        transforms.Compose([
            transforms.Resize([96, 96]),
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(64),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #均值，标准差
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

# 字典生成式 {key: valid for key in [string,string]}2022年12月18日13:01:45
my_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_trainsforms[x]) for x in ['train', 'valid']}
# 合成字典表达式
# loader_lambda = DataLoader(my_datasets[x], batch_sizes= batch_size, shuffle= True)
# data_loaders = {x: lambda for x in ['train', 'valid']}
my_data_loaders = {x: DataLoader(my_datasets[x], batch_size= batch_sizes, shuffle= True) for x in ['train', 'valid']}
data_size = {x: len(my_datasets[x]) for x in ['train', 'valid']}

# 3、构建模型 2022年12月18日13:21:16
my_model = getattr(models, model_name)(pretrained=True)  # 设定模型
for param in my_model.parameters():  # 冻结参数
    param.requires_grad = False
# 设定自己类别
my_model.fc = nn.Linear(my_model.fc.in_features, numclasses)
# 是否训练所有层2022年12月18日13:24:35
params_to_update = my_model.parameters()
# parameters_to_update = []  # 查考需要更新的参数
# for name, params in my_model.named_parameters():
#     if params.requires_grad:
#         parameters_to_update.append(name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 启用GPU
# 4、开始训练
# 优化器设定
my_opt = torch.optim.Adam(params_to_update, lr=1e-2)  # 设定优化器
scheduler = torch.optim.lr_scheduler.StepLR(my_opt, step_size=10, gamma=0.1)  # 优化学习率
my_loss = nn.CrossEntropyLoss()


# 训练过程
def train_model(model, dataloaders, numepochs, optimizer, loss, filename):
    # 设定训练过程的一系列指标
    val_acc_history = []
    train_acc_history = []
    train_loss = []
    valid_loss = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]
    best_acc = 0  # 最好的精度
    best_model_wts = copy.deepcopy(model.state_dict())
    start = time.time()  # 记录开始时间
    model.to(device)  # 加载模型到设备
    for epoch in range(numepochs):
        print('Epoch {}/{}'.format(str(epoch), str(numepochs-1)))  # 打标记
        print('_'*15)
        # 每个epoch要做的事 分训练与验证，每次训练完成进行验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for data, label in dataloaders['train']:
                data = data.to(device)
                label = label.to(device)   # 每个数据都放入设备
                optimizer.zero_grad()  # 梯度清空
                output = model(data)  # 计算输出
                cost = loss(output, label)  # 计算代价
                _, preds = torch.max(output, 1)
                cost.backward()  # 梯度反向求导
                optimizer.step()  # 梯度更新

                running_loss += cost.item() * data.size(0)
                running_corrects += torch.sum(preds == label.data)  # 预测结果最大的和真实值是否一致

            epoch_loss = running_loss / data_size['train']
            epoch_acc = running_corrects.double() / data_size['train']
            time_elapsed = time.time() - start  # 一个epoch我浪费了多少时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # 获取效果最好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer,
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_loss.append(epoch_loss)
                # scheduler.step(epoch_loss)#学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()  # 学习率衰减
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    # return model, val_acc_history, train_acc_history, valid_loss, train_loss, LRs

# 5、验证测试





# 单元测试再推一次测试2022年12月18日12:54:42
if __name__ == '__main__':
    train_model(my_model, my_data_loaders, 20, my_opt, my_loss, my_wight)








