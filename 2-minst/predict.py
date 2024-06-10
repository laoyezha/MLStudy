import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


# 加载图片
image_path = './img/4-1.jpeg'  # 替换为你的图片路径
image = Image.open(image_path)
nImage = image.resize((28, 28))

# 转换为NumPy数组
# image_array = np.array(image)
# 可以转换为灰度图像
gray_image_array = np.array(nImage.convert('L'))
# 需要黑底白字
# gray_image_array = 255 - gray_image_array
# # # print(gray_image_array.shape)  # 打印数组形状
# gray_image_array[gray_image_array<80] = 0
# gray_image_array[gray_image_array>80] = 255
print(gray_image_array)
np.savetxt("./array.txt", gray_image_array, fmt='%d')
plt.imshow(gray_image_array)
plt.show()

model:tf.keras.Sequential = tf.keras.models.load_model('./model')
# model.summary()

# 预测
# gray_image_array = gray_image_array.reshape((1, 28, 28, 1)).astype('float32') / 255
gray_image_array = gray_image_array.reshape((1, 28*28)).astype('float32')
prediction = model.predict(gray_image_array)
print(prediction)
print(np.argmax(prediction))