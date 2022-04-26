import common
import torch


# 使用one-hot提取验证码的文本特征
# 根据验证码中每一个数在随机验证码长度中的下标生成二维特征矩阵
# 说白了就是对一个空白（全零）张量中的指定位置进行赋值（赋 1）
# 指定位置就是指的是验证码每一个数在随机验证码字符串中的下标
def text2vec(text):
    # 教程：https://www.bilibili.com/video/BV1BP4y1b7Er?p=4

    # 验证码长度，和随机验证码长度（0123456789abcdefghijklmnopqrstuvwxyz）
    # 根据验证码长度和随机验证码长度生成一个全为0的矩阵
    # 例 ：
    #   验证码为85a2 长度为4，
    #   随机验证码为 0123456789abcdefghijklmnopqrstuvwxyz 长度为36
    # 那么生成的矩阵就是 4 x 36 全是0的二维数组，36对应的就是随机验证码字符串的长度，因为需要根据下标替换为0
    vectors = torch.zeros((common.captcha_size, common.captcha_array.__len__()))

    # 打印矩阵
    # print("==============生成矩阵================")
    # print(vectors)
    # vectors[0,0] = 1
    # vectors[1,3] = 1
    # vectors[2,4] = 1
    # vectors[3, 1] = 1

    # 根据验证码的每一个数从随机验证码字符串中找出下标
    for i in range(len(text)):
        # 当前验证码在随机验证码字符串中的下标
        index = common.captcha_array.index(text[i])

        # 根据下标把矩阵中的0替换为1，这样就把当前验证码的下标特征放到矩阵中了
        vectors[i][index] = 1

    # print("===============替换下标（特征矩阵）===============")
    # print(vectors)

    return vectors


# 根据上一步生成的验证码特征矩阵反向解析出验证码，提取为1的下标去随机验证码字符串中查找根据下标查找
def vectotext(vec):
    # 取特征矩阵数组中每一维最大值的下标，下标数组
    indexArray = torch.argmax(vec, dim=1)

    # 遍历下标数组，根据每一个下标从随机验证码字符串中查找下标对应的字符
    text_label = ""
    for index in indexArray:
        text_label += common.captcha_array[index]
    return text_label


if __name__ == '__main__':
    # 根据验证码生成特征矩阵
    vec = text2vec("3574")

    # 打印特征矩阵维度 4 x 36
    print(vec.shape)

    # 根据特征矩阵还原字符串
    print(vectotext(vec))

    # 根据验证码生成特征矩阵
    vec = text2vec("3574")

    # 4 x 10
    # 随机验证码字符串：0123456789