import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from forward_noising_lidar import DenoiseDiffusion
from dataloader2 import LIDARHS
from GQHmodel2 import UNet
from simpletest_ddim import SimpleModel
import torch.nn as nn
def setseed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def center_feature(diffusion,HS_out,T,BATCH_SIZE):
    hsnew = diffusion.q_sample(HS_out, torch.tensor([T - 1]).to(HS_out.device))
    with torch.no_grad():
        center_ = []
        for t_ in range(T):
            t = T - t_ - 1
            hsnew, center = diffusion.p_sample(hsnew, hsnew.new_full((BATCH_SIZE,), t, dtype=torch.long))
            center_.append(center)
        center_feature = torch.cat(center_, dim=1)



        # print(center_feature.shape)
    return center_feature[:,-200:]



def main():
    seed = 3407
    setseed(seed)
    epoc = 15
    T = 500
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    data = LIDARHS(32, mode='train')
    train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)
    # 定义与权重文件相匹配的模型结构
    model = UNet(5).to(device)
    # 加载预训练权重文件
    weights_path = 'trained_models/ddpm_mse_epochs_500.pth'
    pretrained_weights = torch.load(weights_path)
    # 将加载的权重加载到模型中
    model.load_state_dict(pretrained_weights)
    # 将模型设置为评估模式（关闭 dropout 和 batch normalization 的训练模式）
    model.eval()
    diffusion = DenoiseDiffusion(eps_model=model, n_steps=T, device=device)

    net = SimpleModel().to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0

    best_loss, best_epoch = 2, 0

    net.train()
    Loss = []
    for epocs in range(epoc):

        for step, (lidar_out, HS_out, lable_out, ik_out) in enumerate(train_loader):
            lidar_out = lidar_out.type(torch.float).to(device)
            HS_out = HS_out.type(torch.float).to(device)
            lable_out = lable_out.type(torch.LongTensor).to(device) - 1
            # new_hs_out = center_feature(diffusion,HS_out,T,BATCH_SIZE)

            hsnew = diffusion.q_sample(HS_out, torch.tensor([T - 1]).to(device))
            with torch.no_grad():
                hsnew, center = diffusion.ddim({0: hsnew, 1: lidar_out}, 50)
                # print(center.shape)

            outputs = net(center)
            loss = criterion(outputs, lable_out)
            # print(new_hs_out.shape)
            # print(loss,'loss')
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if best_loss > loss:
                best_loss = loss
                best_epoch = epocs
                torch.save(net.state_dict(), 'trained_models/netfordif_weights_ddim.pth')

        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epocs + 1,
                                                                         total_loss / (epocs + 1),
                                                                         loss.item()))
        Loss.append(loss.item())
    print("best_loss:", best_loss, "best_epoch:", best_epoch)

if __name__ == '__main__':
    main()