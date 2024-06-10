import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image = tf.keras.preprocessing.image

# 读取图片
img = image.load_img('./img/8-2.jpeg', color_mode='grayscale', target_size=(28, 28))

# 转换为numpy数组并归一化
img_array = image.img_to_array(img)
img_array = 255-img_array
print(img_array)
plt.imshow(img_array)
plt.show()
# 扩展维度以符合模型输入要求（因为模型期望的输入是(batch_size, height, width, channels)）
gray_image_array = np.expand_dims(img_array, axis=0)


model:tf.keras.Sequential = tf.keras.models.load_model('./model')
# model.summary()

# 预测
# gray_image_array = gray_image_array.reshape((1, 28, 28, 1)).astype('float32')
gray_image_array = gray_image_array.reshape((1, 28*28)).astype('float32')
prediction = model.predict(gray_image_array)
print(prediction)
print(np.argmax(prediction))