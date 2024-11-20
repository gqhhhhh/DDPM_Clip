import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import random
from matplotlib import pyplot as plt

def applyPCA(X, numComponents = 5):
    num_pixels =X.shape[0] * X.shape[1]
    num_bands = X.shape[2]
    data_2d = X.reshape(num_pixels, num_bands)
    data_2d_std = (data_2d - np.mean(data_2d, axis=0)) / np.std(data_2d, axis=0)
    pca = PCA(numComponents)
    pca.fit(data_2d_std)
    components = pca.components_
    projected = pca.transform(data_2d_std)
    projected_3d = projected.reshape(X.shape[0], X.shape[1], numComponents)
    # print(projected_3d.shape)
    # 显示每个主成分的图像
    # for i in range(3):
    #     plt.subplot(1, 4, i + 1)
    #     plt.imshow(projected_3d[:, :, i], cmap='gray')
    #     plt.title(f'Principal Component {i + 1}')
    #     plt.axis('off')
    # plt.show()
    return projected_3d

class LIDARHS(Dataset):
    def __init__(self, patchsize, mode = 'train', classnum = 100):
        self.mode = mode
        # self.padding_layers = int((patchsize - 1)/2) # 给边边补个0
        self.padding_layers = int(patchsize /2) # 给边边补个0
        self.lidar_mat = np.squeeze(sio.loadmat('data/LIDAR.mat')['LiDAR'].astype(np.float32))
        # self.lidar_mat = np.pad(self.lidar_mat, self.padding_layers, mode='constant', constant_values=0)
        self.lidar_mat = np.pad(self.lidar_mat, self.padding_layers, mode='symmetric')
        self.HS_mat = np.squeeze(sio.loadmat('data/2012_Huston.mat')['spectraldata'].astype(np.float32))
        self.HS_mat = applyPCA(self.HS_mat, 8)
        # self.HS_mat = np.pad(self.HS_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
        #                 mode='constant', constant_values=0)
        self.HS_mat = np.pad(self.HS_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
                        mode='symmetric')
        # plt.imshow(self.HS_mat[:, :,10], cmap='gray')
        # plt.show()
        self.train_lable_os = 'data/trainlables.mat'
        self.test_lable_os = 'data/testlables.mat'
        # if self.mode == 'train':
        #     self.lable_mat = np.squeeze(sio.loadmat(self.train_lable_os)['trainlabels'].astype(np.float32))
        # else:
        self.lable_mat = np.squeeze(sio.loadmat(self.test_lable_os)['testlabels'].astype(np.float32))
        self.lidar = []
        self.HS = []
        self.lable = []
        self.ik = []
        for i in range(0, len(self.lable_mat)):
            for k in range(0, len(self.lable_mat[0])):
                ik_now = (i,k)
                lable_now = self.lable_mat[i][k]
                if lable_now != 0:
                    self.lable.append(lable_now)
                    # lidar_now = self.lidar_mat[i:i + (self.padding_layers*2+1), k:k + (self.padding_layers*2+1)]
                    lidar_now = self.lidar_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]

                    lidar_now = np.expand_dims(lidar_now, axis=0)
                    # HS_now = self.HS_mat[i:i + (self.padding_layers*2+1), k:k + (self.padding_layers*2+1)]
                    HS_now = self.HS_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]

                    HS_now = np.transpose(HS_now, (2, 0, 1))
                    self.lidar.append(lidar_now)
                    self.HS.append(HS_now)
                    self.ik.append(ik_now)

        # print(np.array(self.lidar).shape,np.array(self.HS).shape,np.array(self.lable).shape)
        # (2832, 1, 17, 17)(2832, 30, 17, 17)(2832, )

        self.train_hsi = []
        self.train_lidar = []
        self.train_lable = []
        self.train_ik = []
        for cls in range(1,int(max(self.lable)) + 1):
            indices = [index for index, value in enumerate(self.lable) if value == cls]
            random_indices = random.sample(indices, classnum)
            self.train_hsi += [self.HS[index] for index in random_indices]
            self.train_lidar += [self.lidar[index] for index in random_indices]
            self.train_lable += [self.lable[index] for index in random_indices]
            self.train_ik += [self.ik[index] for index in random_indices]
        # print(np.array(self.train_hsi).shape,np.array(self.train_lidar).shape,np.array(self.train_lable).shape)
        # print(self.train_lable)
        # (450, 30, 17, 17)(450, 1, 17, 17)(450, )
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_lable)
        else:
            return len(self.lable)
    def __getitem__(self, item):
        if self.mode == 'train':
            lidar, HS, lable, ik = self.train_lidar[item], self.train_hsi[item], self.train_lable[item], self.train_ik[item]
        else:lidar, HS, lable, ik = self.lidar[item], self.HS[item], self.lable[item], self.ik[item]
        # print(max(self.lable))

        # lables = torch.zeros(15)
        # lables[int(lable) - 1] = 1.0
        return lidar, HS, lable, ik


if __name__ == '__main__':

    data = LIDARHS(48, mode='train')
    train_loader = DataLoader(data, batch_size = 3, shuffle = True, num_workers = 1)
    device = 'cpu'

    for step, (lidar_out, HS_out, lable_out, ik_out) in enumerate(train_loader):
        lidar_out = lidar_out.type(torch.float).to(device)
        HS_out = HS_out.type(torch.float).to(device)
        lable_out = lable_out.type(torch.LongTensor).to(device) - 1
        # i = ik_out[:][0]
        # k = ik_out[:][1]
        # if step == 1:
        #     print(lable_out[0])
        #     print(ik_out)
        #     print(i[0])
        #     print(k[0])
        #     print(lidar_out[0])
        # 例如8*8patch，偶数时中心点在[4,4]的位置

        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * c * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *

        if step == 1:
            from matplotlib import pyplot as plt
            print(lable_out,type(lable_out),lable_out.shape)
            print(lidar_out.shape)
            print(HS_out.shape)
            # print(HS_out[0][1][ :, :])
            # print(lable_out[0])
            plt.subplot(1, 2, 1)
            plt.imshow(HS_out[0][1][ :, :], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(lidar_out[0][0], cmap='gray')
            plt.show()
