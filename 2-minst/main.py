import os
import gzip
import numpy as np
import tensorflow as tf

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前16个字节的头部信息
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 将数据转换为28x28的图像数组
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前8个字节的头部信息
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 设置数据集文件的路径
datasetPath = './dataset/MNIST_data/'
trainImgPath = os.path.join(datasetPath, 'train-images-idx3-ubyte.gz')
trainLabelPath = os.path.join(datasetPath, 'train-labels-idx1-ubyte.gz')
ValidImgPath = os.path.join(datasetPath, 't10k-images-idx3-ubyte.gz')
ValidLabelPath = os.path.join(datasetPath, 't10k-labels-idx1-ubyte.gz')
# 加载图像和标签数据
x_train = load_mnist_images(trainImgPath)
y_train = load_mnist_labels(trainImgPath)
x_valid = load_mnist_images(ValidImgPath)
y_valid = load_mnist_labels(ValidLabelPath)

print(x_train.shape)
