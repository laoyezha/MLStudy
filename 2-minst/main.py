import os
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist_images(filename) -> np.ndarray:
    with gzip.open(filename, 'rb') as f:
        # 跳过前16个字节的头部信息
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 将数据转换为28x28的图像数组
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename) -> np.ndarray:
    with gzip.open(filename, 'rb') as f:
        # 跳过前8个字节的头部信息
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# 设置数据集文件的路径
datasetPath = './dataset/MNIST_data/'
# 加载图像和标签数据
x_train = load_mnist_images(os.path.join(datasetPath, 'train-images-idx3-ubyte.gz'))
y_train = load_mnist_labels(os.path.join(datasetPath, 'train-labels-idx1-ubyte.gz'))
x_valid = load_mnist_images(os.path.join(datasetPath, 't10k-images-idx3-ubyte.gz'))
y_valid = load_mnist_labels(os.path.join(datasetPath, 't10k-labels-idx1-ubyte.gz'))

# 展示数据
# plt.imshow(x_train[1])
# print(x_valid.shape)
# plt.show()


# 创建全连接层
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # 用于将输入的多维数据展平成一维
# model.add(tf.keras.layers.Flatten()) # 用于将输入的多维数据展平成一维
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.03)))
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.03)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), #一种结合了动量和自适应学习率的优化算法
              loss=tf.losses.SparseCategoricalCrossentropy(), # 用于整数标签的多类别分类问题交叉熵
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()] #稀疏准确率，适用于稀疏标签的类别分类问题
              )
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid, y_valid))
# model.fit(x_train, y_train, validation_split=0.25, epochs=100, batch_size=64)

model.evaluate(x_valid,y_valid)

model.save('./model')
