import numpy as np


def oneD_Fourier(data):
    """
    一维傅里叶变换
    """

    # 数据多了一维
    # data = np.squeeze(data)
    # print(data.shape)
    for layer in range(data.shape[0]):
        data[layer] = abs(np.fft.fft(data[layer]))
    # data = data.reshape(-1, data.shape[1], 1)

    return data

