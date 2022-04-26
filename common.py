import random
import time

# 验证码在包含的数字字母
captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
# captcha_array = list("0123456789")
# # 验证码长度
captcha_size = 4
from captcha.image import ImageCaptcha

if __name__ == '__main__':
    print(captcha_array)
    image = ImageCaptcha()
    for i in range(100):
        image_val = "".join(random.sample(captcha_array, 4))

        image_name = "./dataset/test/{}_{}.png".format(image_val, int(time.time()))
        print(image_name)
        image.write(image_val, image_name)
