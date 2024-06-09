import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import datetime
from sklearn import preprocessing

warnings.filterwarnings('ignore')
# %matplotlib inline
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

keras = tf.keras

features = pd.read_csv('./temps.csv')
# print(features.head())
# print(features.shape)

years = features['year']
months = features['month']
days = features['day']
# 合并三个数组为一个
times = np.column_stack((years,months,days))
# times = zip(years,months,days)

dates = list(map(lambda ts:datetime.datetime(ts[0],ts[1],ts[2]), times))
# print(dates)


# 数据展示
def showData(dates, features):
    plt.style.use('fivethirtyeight')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('日期')
    ax1.set_ylabel('温度')
    ax1.set_title('实际')

    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('日期')
    ax2.set_ylabel('温度')
    ax2.set_title('昨天')

    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('日期')
    ax3.set_ylabel('温度')
    ax3.set_title('前天')

    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('日期')
    ax4.set_ylabel('温度')
    ax4.set_title('搞笑')

    plt.tight_layout(pad=2)
    plt.show()


# 热编码
features = pd.get_dummies(features)
# print(features.head(5))

# 提取actual列的数据
labels = np.array(features['actual'])

# axis=0 通常意味着操作是按列进行的，比如计算每列的和或平均值。
# axis=1 通常意味着操作是按行进行的，比如计算每行的和或平均值
# 删除actual列
features_2 = features.drop('actual', axis=1)
# print(features_2.head(5))

# 提取列名
cloNames = features_2.columns

featuresDatas = np.array(features_2)

standardDatas = preprocessing.StandardScaler().fit_transform(featuresDatas)
# print(standardDatas[:10])
# print(labels.shape)


# 构建模型
model = keras.Sequential()
# kernel_initializer='random_normal'设置神经网络权重的初始化方法,random_normal从具有给定均值的正态分布中随机初始化权重
# kernel_regularizer 指定应用于神经网络权重的正则化函数。通过向损失函数添加一个惩罚项来实现
model.add(keras.layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.L2(0.03))) # 第一层16个神经元
model.add(keras.layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.L2(0.03))) # 第二层32个神经元
model.add(keras.layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.L2(0.03))) # 输出层输出1个结果

# SGD ：随机梯度下降是最基础的优化算法
# 均方误差（Mean Squared Error, MSE）
model.compile(optimizer=keras.optimizers.SGD(0.001), loss='mean_squared_error')

# 训练
# validation_split=0.25 : 指定所有数据的25%为验证集
# batch_size: 指定每个批次的样本数。一个批次中的样本会一次性通过模型进行训练，从而计算一次梯度下降更新。较大的批次可以加速训练，但会消耗更多的内存
# epochs: 指定训练过程中的迭代次数。每个 epoch 指的是遍历整个训练集一次的过程。训练将在达到指定的 epoch 数目后停止
model.fit(standardDatas, labels, validation_split=0.25, epochs=100, batch_size=64)

# 显示一个模型的结构 summary（概要），它会展示模型的各个层级（layer）、每层的输出形状、参数数量等信息。
# model.summary()

predict = model.predict(standardDatas)
# print(predict.shape)


plt.plot(dates, labels, 'b-', label = '实际')
plt.plot(dates, predict, 'ro', label = '预测')
plt.xticks(rotation=60)
plt.legend() # 添加图例

plt.show()