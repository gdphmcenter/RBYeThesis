# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from datafft import oneD_Fourier
from datanormalization import oneD_Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder


#
# for faultD in faultData:
#     Data.vstack(np.array(faultD))
#
# print(Data.shape)

# #---------------------------------------------
# # 将X/Y数组分为故障数据和正常数据
# #--------------------------------------------
# #使用Y中的第二列获取数组索引以分割正常数据和故障数据
# fault_indices, normal_indices=np.where(Y[:,1] == 'Fault')[0], np.where(Y[:,1] == 'Run')[0]
# #使用索引将正常数据分配给“Xnormal ”,将故障数据分配给“Xanomaly”
# Xnormal, Xanomaly = X[normal_indices,:,:], X[fault_indices,:,:]
# #使用索引将正常数据的标签指定为“Ynormal ”,将故障标签指定为“Yanomaly”
# Ynormal, Yanomaly = Y[normal_indices,:], Y[fault_indices,:]   #标签有:索引、状态、类型


def loadDataset(normalDataLenth=150, feature_index=13):
    '''
    :param normalDataLenth: 正常数据集的长度
    :param feature_index: A-flux 磁通波形(用于绘图)。使用0表示A+IGBT-I，使用1表示A+*IGBT-I，依此类推。
    :return: 数据和标签
    '''

    features = ['A+IGBT-I', 'A+*IGBT-I', 'B+IGBT-I', 'B+*IGBT-I', 'C+IGBT-I', 'C+*IGBT-I', 'A-FLUX',
                'B-FLUX', 'C-FLUX', 'MOD-V', 'MOD-I', 'CB-I', 'CB-V', 'DV/DT']  # 用作标题的波形名称

    system = 'SCL'  # 选择一个系统进行加载和绘图。选择RFQ、DTL、CCL或SCL
    # ----------------------------------------------------------------------------------------------
    # 加载波形(X)和标签(Y)数据集，例如RFQ、DTL、CCL、SCL
    # ---------------------------------------------------------------------------------------------

    X = np.load('./%s.npy' % system)  # ---> x具有形状:(脉冲、时间、特征)
    Y = np.load('./%s_labels.npy' % system, allow_pickle=True)  # ---> y数组有形状:(脉冲，标签)->标签有:索引，状态，类型

    tmpSet = set()
    for i in Y[:, 2]:
        tmpSet.add(i)

    # ------------------------------------------------------------------------ # Data

    useingList = [
        'IGBT B+* Low Fault',
        'C FLUX Low Fault',
        'DV/DT High Fault',
        '- CB V Low Fault',
        # 'MOD I High Fault',
        # '- CB I High Fault'
    ]

    fault_indicesList = []

    tmpSet = list(tmpSet)

    for faultName in useingList:
        fault_indicesList.append(np.where(Y[:, 2] == faultName)[0])

    normal_indices = np.where(Y[:, 2] == 'Normal')[0]

    faultData = []
    normalData = []

    for faultIndex in fault_indicesList:
        faultData.append(X[faultIndex, :, feature_index])
    normalData.append(X[normal_indices, :, feature_index])

    # print(np.array(normalData[0]).shape)
    # for idx, i in enumerate(faultData):
    #     print(np.array(i).shape)
    #     print(tmpSet[idx])

    Data = np.array(normalData[0])[0:normalDataLenth, :]
    for i in faultData:
        Data = np.vstack([Data, np.array(i)])

    print(Data.shape)

    # ------------------------------------------------------------------------ # label

    faultLabel = []
    normalLabel = []

    for faultIndex in fault_indicesList:
        faultLabel.append(Y[faultIndex, 2])
    normalLabel.append(Y[normal_indices, 2])

    Label = np.array(normalLabel[0])[0:normalDataLenth]
    for i in faultLabel:
        Label = np.hstack([Label, np.array(i)])

    print(Label)

    Data = oneD_Normalize(Data)
    # Data = oneD_Fourier(Data)

    return Data, Label


class Mydataset(Dataset):
    def __init__(self, Train=True, normalDataLenth=150):
        data, label = loadDataset(normalDataLenth=normalDataLenth)
        encoder_label = LabelEncoder()
        label = encoder_label.fit_transform(label)
        if Train:
            self.data = data[:normalDataLenth]
            self.label = label[:normalDataLenth]

        else:
            self.data = data
            self.label = label

    def __getitem__(self, item):  # overwrite
        # 将numpy数据转为tensor
        data = torch.from_numpy(self.data[item])  # 转为tensor

        return data, self.label[item]  # 返回一个数据和一个label

    def __len__(self):  # 返回数据集的长度
        return len(self.label)


if __name__ == '__main__':
    # data, label = loadDataset(normalDataLenth=150)
    batch_size = 90
    train_loader = torch.utils.data.DataLoader(
        Mydataset(),
        batch_size=batch_size, shuffle=True)

    for (data, target) in train_loader:
        print(data.shape)

    print("finish")
