import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import my_datasets          # 引入my_datasets文件，自定义加载数据集
from Net import Network     # 引入Net文件中的Network类，卷积神经网络
from torch.utils.tensorboard import SummaryWriter



# 根据是否有GPU选择是否在GPU上运行，默认是在CPU运行
# 数据集和网络使用to加载
# https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
# https://blog.csdn.net/qq_38832757/article/details/113630383
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # 导入数据集 训练集和测试集
    # 自定义的mydatasets导入类
    train_datas = my_datasets.mydatasets(r"/content/drive/MyDrive/captcha_ocr-main/dataset/test")
    test_data = my_datasets.mydatasets(r"/content/drive/MyDrive/captcha_ocr-main/dataset/test")
    train_dataloader = DataLoader(train_datas, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    print("=======数据集========")
    # 遍历数据集，打印信息
    for i, (imgs, targets) in enumerate(train_dataloader):
        print(imgs.shape)

        # 打印3行，遍历了3次，数据集中有95个图片，batch_size大小为32，所以一次加载32个图片，加载了3次
        #   32：batch_size大小（一次加载32个图片）
        #   1：灰度单通道，3是RGB
        #   72：图片大小（矩阵） 72 x 270
        #   270：图片大小（矩阵） 72 x 270
        # torch.Size([32, 1, 72, 270])
        # torch.Size([32, 1, 72, 270])
        # torch.Size([31, 1, 72, 270])


    # 创建网络  to选择是否在GPU上运行
    net = Network().to(device)

    # 创建损失函数 熵损失
    loss_fn = nn.MultiLabelSoftMarginLoss().to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)

    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MultiLabelSoftMarginLoss().cuda()

    # 创建优化器,减少损失值loss
    # 根据反向传播算法更新神经网络中的参数，以达到降低损失值loss的目的
    # https://blog.csdn.net/Carlsummer/article/details/119774752
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(net.parameters(),lr = 0.01, momentum=0.3)


    total_step = 0

# 迭代轮数
for i in range(30):
    # 遍历数据集
    # imgs 和 lables就是__getitem__方法返回的图片的张量和验证码结果的特征矩阵
    for j, (imgs, lables) in enumerate(train_dataloader):
        # 选择是否在GPU上运行
        imgs = imgs.to(device)
        lables = lables.to(device)

        # 调用网络训练，框架会调用网络的forward方法
        outputs = net(imgs)

        # 熵损失
        loss = loss_fn(outputs, lables)

        # 将梯度清零
        # 调用backward()函数之前都要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加
        optimizer.zero_grad()

        # 计算梯度
        loss.backward()

        # 计算梯度后更新参数
        optimizer.step()

        # 每10
        if j % 10 == 0:
            total_step += 1
            # loss损失值：和正确值之间的差距
            print("轮{},训练{}次,loss:{}".format(i,total_step, loss.item()))

# 保存
# torch.save(net, "model.pth")
model = DataParallel(net)
real_model = model.module
torch.save(real_model,"model.pth")
