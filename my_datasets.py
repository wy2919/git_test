import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import one_hot


# 创建自定义数据集加载类 继承官方类
class mydatasets(Dataset):

    def __init__(self, root_dir):
        super(mydatasets, self).__init__()

        # 根据数据集文件夹路径把所有图片路径加载到列表
        self.list_image_path = [os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]

        # 统一数据集规格
        self.transforms=transforms.Compose([
            # 统一数据集宽高
            transforms.Resize((60,160)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])

        # 通道报错改为这个
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),  # 转化为pytorch中的tensor
        #     transforms.Lambda(lambda x: x.repeat(1, 1, 1)),  # 由于图片是单通道的，所以重叠三张图像，获得一个三通道的数据
        # ])

    # 传入下标，我们根据下标返回该数据集图片的：图片特征矩阵和验证码结果特征矩阵
    def __getitem__(self, index):
        # 根据传入的下标从初始化类时加载的数据集集合中获取下标对应的数据集路径
        image_path = self.list_image_path[index]
        # print(image_path)
        # 打开图片文件
        img_ = Image.open(image_path)

        # 获取到图片名称
        image_name = image_path.split("/")[-1]
        # print(image_name)
        # linux下获取文件名称 [-1]获取最后
        # image_name = image_path.split("\\")[-1]

        # 图片格式数据统一化，避免大小不一样等，返回pytorch可识别的数据格式（张量）
        img_tesor = self.transforms(img_)

        # 取出名称中的验证码结果：a73g_159070064801343219992907989638805681832.png
        # 结果img_lable就是a73g
        img_lable = image_name.split("_")[0]

        # 调用one_hot.py的方法对验证码结果字符串提取特征矩阵
        img_lable = one_hot.text2vec(img_lable)


        # 对特征矩阵二维数组展平，展平为一维数组，也就是二维数组相称 4 x 36 = 144
        img_lable = img_lable.view(1, -1)[0]

        # 返回图片处理后的格式，和验证码特征矩阵
        return img_tesor, img_lable

    def __len__(self):
        return self.list_image_path.__len__()


# 测试
if __name__ == '__main__':
    # 传入数据集文件夹路径
    d = mydatasets(r"D:\Python\pycharm\machine_learning\captcha_ocr-main\captcha_ocr-main\dataset\test")

    # 根据下标0获取单个图片，框架自动调用__getitem__方法
    img, label = d[0]

    # 打印图片（张量）
    # torch.Size([1, 60, 160])
    #    1是灰度 3是RGB
    #    60是高
    #    160是宽
    print(img.shape)

    # 打印验证码特征矩阵（扁平后的1维数组）
    print(label.shape)
