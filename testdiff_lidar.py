import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from forward_noising_lidar import DenoiseDiffusion
from dataloader2 import LIDARHS
from GQHmodel2 import UNet
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

def setseed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # seed = 34
    setseed(3407)
    T = 500
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    data = LIDARHS(32, mode='test')
    test_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # 定义与权重文件相匹配的模型结构
    model = UNet(5).to(device)
    # 加载预训练权重文件
    weights_path = 'trained_models/ddpm_mse_epochs_500.pth'
    pretrained_weights = torch.load(weights_path)
    # 将加载的权重加载到模型中
    model.load_state_dict(pretrained_weights)
    # 将模型设置为评估模式（关闭 dropout 和 batch normalization 的训练模式）
    model.eval()
    diffusion = DenoiseDiffusion(eps_model=model,n_steps=T,device=device)

    target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
        , 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway',
                    'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']

    for step, (lidar_out, HS_out, lable_out, ik_out) in enumerate(test_loader):
        # if step%10!=0:continue
        lidar_out = lidar_out.type(torch.float).to(device)
        HS_out = HS_out.type(torch.float).to(device)
        lable_out = lable_out.type(torch.LongTensor).to(device) - 1

        for m in range(3):
            plt.subplot(3, 6, 1+m*6)
            plt.imshow(HS_out.cpu()[m+7][2][:, :], cmap='gray')
            plt.title(target_names[lable_out.cpu()[m+7].item()],{'size':7})

        HS_out.to(device)
        hsnew = diffusion.q_sample(HS_out, torch.tensor([T - 1]).to(device))
        with torch.no_grad():
            i = 2%6
            center_ = []
            time_1 = time.perf_counter()
            for t_ in range(T):
                t = T - t_ - 1
                hsnew, center = diffusion.p_sample({0:hsnew,1:lidar_out}, hsnew.new_full((BATCH_SIZE,), t, dtype=torch.long))
                # print(t,end=' ')
                hscpu = hsnew.cpu()

                # if t%50==0:print('\n')
                center_.append(center)

                if t % 100 == 0:

                    plt.subplot(3, 6,i)
                    plt.imshow(hscpu[7][2][:, :], cmap='gray')
                    plt.title(f'time {t}',{'size':7})
                    plt.subplot(3, 6, 6+i)
                    plt.imshow(hscpu[8][2][:, :], cmap='gray')
                    plt.title(f'time {t}', {'size': 7})
                    plt.subplot(3, 6,12+ i)
                    plt.imshow(hscpu[9][2][:, :], cmap='gray')
                    plt.title(f'time {t}', {'size': 7})
                    plt.axis('off')
                    i += 1
            time_2 = time.perf_counter()
            print(f'time taken: {time_2-time_1}')
            center_feature = torch.cat(center_,dim=1)
            print(center_feature.shape)
            # plt.show()
            plt.savefig(f'fig/figtrain500lidar{step}')
