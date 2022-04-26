import torch
from torch import nn
import common


# 创建网络 继承nn.Module
class Network(nn.Module):
    # 初始化层
    def __init__(self):
        super(Network, self).__init__()

        # =================卷积层====================

        # 参数是自己计算的

        # layer1卷积层前：[32, 1, 60, 160]
        # 32(batch_size大小,多少张图片) 1(通道数1是灰度 3是RGB)
        # 60(图片矩阵高) 160(图片矩阵宽)
        self.layer1 = nn.Sequential(
            # in_channels=1（1灰度 3RGB）  out_channels=64（输出 过滤器数量 也就是特征图数量(矩阵) 也是下一个卷积层的输入）
            # kernel_size=3（卷积核步长）  padding=1（周围填充）
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1, padding=1),
            # 非线性激活函数
            nn.ReLU(),
            # 池化 减小特征矩阵体积 宽高除2
            nn.MaxPool2d(kernel_size=2)
        )
        # layer1卷积层后：[32, 64, 30, 80] === 32张图片，64个特征图（矩阵），每个特征图大小为：30 x 80
        # 32张不变，64是特征图数量layer1设置的，宽高60, 160由于池化层体积缩小一半变为30, 80 不为整数向下取整

        # layer2卷积层前：[32, 64, 30, 80]  也就是卷积层layer1的输出
        self.layer2 = nn.Sequential(
            # in_channels=64（是layer1的输出）out_channels=128（layer2的输出）
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # layer2卷积层后：[32, 128, 15, 40] === 32张图片，128个特征图（矩阵），每个特征图大小为：15 x 40

        # layer3卷积层前：[32, 128, 15, 40]  也就是卷积层layer2的输出
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 256, 7, 20]
        )
        # layer3卷积层后：[32, 256, 7, 20] === 32张图片，256个特征图（矩阵），每个特征图大小为：7 x 20 不为整数向下取整

        # layer4卷积层前：[32, 256, 7, 20]  也就是卷积层layer3的输出
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 512, 3, 10]
        )
        # layer4卷积层后：[32, 512, 3, 10] === 32张图片，512个特征图（矩阵），每个特征图大小为：3 x 10不为整数向下取整
        # 卷积层越多提取到的特征图越多，但是每个特征图的大小越小，特征更细化，每个图片都将提取到512个特征图

        # 将四维张量转换为二维张量之后，才能作为全连接层的输入

        # 全连接层前：[32, 512, 3, 10]  也就是卷积层layer4的输出
        self.layer6 = nn.Sequential(
            # 先展平，展平为一维
            # 展平前：[32, 512, 3, 10]
            # 展平后：[32, 15360]  也就是 512 x 3 x 10  32张15360一维的特征
            nn.Flatten(),

            # 全连接层
            # in_features=15360是 nn.Flatten()结果的[32, 15360] 15360 也就是512 x 3 x 10
            # out_features=4096 每个矩阵输出的特征数量
            nn.Linear(in_features=30720, out_features=4000),

            # 减少 30% 的神经元防止过拟合
            nn.Dropout(0.3),
            # 激活
            nn.ReLU(),

            # 全连接层
            # in_features=4096 上一个全连接层的输出
            # out_features每个矩阵输出的特征数量
            #       是全部类别（验证码长度 x 随机验证码字符长度） 这里是：4 x 36 4(验证码长度) 36(0123456789abcdefghijklmnopqrstuvwxyz)
            nn.Linear(in_features=4000, out_features=common.captcha_size * common.captcha_array.__len__())
        )

    # 前向传播
    def forward(self, x):
        # 调用卷积层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 将四维张量转换为二维张量之后，才能作为全连接层的输入
        x = x.view(x.size(0), -1)

        # 调用全连接层
        x = self.layer6(x)
        return x

    @staticmethod
    def get_device(gpu_id):
        if gpu_id == -1:
            device = torch.device('cpu'.format(str(gpu_id)))
        else:
            device = torch.device('cuda:{}'.format(str(gpu_id)))
        return device

if __name__ == '__main__':
    # 生成一个模拟数据
    #   15：图片张数（batch_size的大小）batch_size大小为15，所以一次加载15个图片
    #   1：图片灰度 3是RGB和上面的初始卷积层输入一致
    #   72：图片大小，矩阵
    #   270：图片大小，矩阵
    data = torch.ones(15, 1, 60, 160)
    model = Network()
    model = model.to(Network.get_device(-1))
    # model = model.to(Network.get_device(0))

    x = model(data)
    print(x.shape)
    print(torch.cuda.is_available())