import os
from os.path import split
import numpy as np
from matplotlib import pyplot as plt

# 读取天气数据
data_dir = '/users/Ryu/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

# 输出一行样例数据
y = lines[0].split(',')
for i, x in enumerate(header):
    print(x, end=':\t\t')
    print(y[i])

# 解析数据为矩阵（时间列不在矩阵中）
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# 选取温度样本列并作图
temp = float_data[:, 1]  # 温度（单位：摄氏度）
plt.plot(range(len(temp)), temp)
plt.show()

# 将前20w个数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    """生成时间序列样本及其目标的生成器

    Args:
        data (list): ：浮点数数据组成的原始数组
        lookback (int): 输入数据应该包括过去多少个时间步
        delay (int): 目标应该在未来多少个时间步之后
        min_index (int): data 数组中的索引，用于界定需要抽取哪些时间步
        max_index (int): data 数组中的索引，用于界定需要抽取哪些时间步
        shuffle (bool, optional): 是打乱样本，还是按顺序抽取样本. Defaults to False.
        batch_size (int, optional): 每个批量的样本数. Defaults to 128.
        step (int, optional): 数据采样的周期. Defaults to 6.

    Yields:
        turple: 一个元组 (samples, targets)，其中 samples 是输入数据的一个批量，targets 是对应的目标温度数组
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
