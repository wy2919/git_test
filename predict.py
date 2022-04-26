from PIL import Image
from torch.utils.data import DataLoader
import one_hot
from Net import Network
import torch
import common
import my_datasets
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_pred():
    m = torch.load("model.pth")
    # m = torch.load("model.pth").cuda()
    m.eval()
    test_data = my_datasets.mydatasets(r"D:\Python\pycharm\免费验证码识别\带带弟弟\img\租号玩")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    correct = 0;
    for i, (imgs, lables) in enumerate(test_dataloader):
        imgs = imgs
        # imgs = imgs.cuda()
        # lables = lables.cuda()
        lables = lables

        lables = lables.view(-1, common.captcha_array.__len__())

        lables_text = one_hot.vectotext(lables)
        predict_outputs = m(imgs)
        predict_outputs = predict_outputs.view(-1, common.captcha_array.__len__())
        predict_labels = one_hot.vectotext(predict_outputs)
        if predict_labels == lables_text:
            correct += 1
            print("预测正确：正确值:{},预测值:{}".format(lables_text, predict_labels))
        else:
            print("预测失败:正确值:{},预测值:{}".format(lables_text, predict_labels))
        # m(imgs)
    print("正确率{}".format(correct / test_length * 100))
def pred_pic(pic_path):
    img=Image.open(pic_path)
    tersor_img=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60,160)),
        transforms.ToTensor()
    ])
    img=tersor_img(img)
    # img=tersor_img(img).cuda()
    print(img.shape)
    img=torch.reshape(img,(-1,1,60,160))
    print(img.shape)
    m = torch.load("model.pth", map_location=lambda storage, loc: storage)
    # m = torch.load("model.pth").cuda()
    outputs = m(img)
    outputs=outputs.view(-1,len(common.captcha_array))
    outputs_lable=one_hot.vectotext(outputs)
    print(outputs_lable)


if __name__ == '__main__':
    # test_pred();
    pred_pic(r"D:\Python\pycharm\machine_learning\captcha_ocr-main\captcha_ocr-main\dataset\test\3rhc_1650808090.png")
    # pred_pic(r"D:\Python\pycharm\免费验证码识别\带带弟弟\img\租号玩\8132_34083593426249801658634887525465009602.png")

