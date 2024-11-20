import random

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from torch.utils.data import DataLoader
from forward_noising_lidar import DenoiseDiffusion
from dataloader2 import LIDARHS
from GQHmodel2 import UNet
from clip_model import Model_All


def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
        , 'Soil', 'Water', 'Residential','Commercial', 'Road', 'Highway', 'Railway',
            'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
    classification = classification_report(y_test, y_pred_test, digits=2, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

def center_feature(diffusion,HS_out,T,BATCH_SIZE):
    hsnew = diffusion.q_sample(HS_out, torch.tensor([T - 1]).to(HS_out.device))
    with torch.no_grad():
        i = 1
        center_ = []
        for t_ in range(T):
            t = T - t_ - 1
            hsnew, center = diffusion.p_sample(hsnew, hsnew.new_full((BATCH_SIZE,), t, dtype=torch.long))
            center_.append(center)
        center_feature = torch.cat(center_, dim=1)
    return center_feature

def setseed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def item_to_color(item):
    if item == 0:
        y = np.array([110, 50, 3]) / 255.
    if item == 1:
        y = np.array([147, 67, 46]) / 255.
    if item == 2:
        y = np.array([0, 0, 255]) / 255.
    if item == 3:
        y = np.array([255, 100, 0]) / 255.
    if item == 4:
        y = np.array([0, 255, 123]) / 255.
    if item == 5:
        y = np.array([164, 75, 155]) / 255.
    if item == 6:
        y = np.array([101, 174, 255]) / 255.
    if item == 7:
        y = np.array([118, 254, 172]) / 255.
    if item == 8:
        y = np.array([60, 91, 112]) / 255.
    if item == 9:
        y = np.array([255, 255, 0]) / 255.
    if item == 10:
        y = np.array([255, 255, 125]) / 255.
    if item == 11:
        y = np.array([255, 0, 255]) / 255.
    if item == 12:
        y = np.array([100, 0, 255]) / 255.
    if item == 13:
        y = np.array([0, 172, 254]) / 255.
    if item == 14:
        y = np.array([0, 255, 0]) / 255.
    return y
def classification_map(map, save_path):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path)
    return 0


def main():
    seed = 3407
    setseed(seed)
    T = 500
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.squeeze(sio.loadmat('data/testlables.mat')['testlabels'].astype(np.float32))
    text = ['Healthy grass.The lush green grass, vibrant and thriving, represents optimal health and vitality.',
     'Stressed grass.The grass, showing signs of stress such as discoloration or wilting, reflects environmental pressures impacting its growth.',
     'Synthetic grass.Artificial grass, made of synthetic fibers, offers a low-maintenance alternative to natural grass, often used in landscaping or sports fields.',
     'Trees.Towering and majestic, trees are essential components of ecosystems, providing oxygen, shade, and habitat for various species.',
     'Soil.The fertile soil, rich in nutrients and organic matter, serves as the foundation for healthy plant growth and biodiversity.',
     'Water.The life-sustaining element, water, is essential for all living organisms, supporting ecosystems and human activities.',
     'Residential.Areas designated for housing, characterized by a mix of single-family homes, apartments, and community amenities.',
     'Commercial.Zones primarily used for business activities, encompassing offices, stores, and commercial enterprises.',
     'Road.The paved pathway for vehicular transportation, connecting destinations and facilitating travel.',
     'Highway.A major road designed for high-speed traffic over long distances, linking cities and regions.',
     'Railway.The railway network, consisting of tracks and trains, enables efficient land-based transportation of goods and passengers.',
     'Parking Lot 1. A designated area for parking vehicles, typically near a specific location or facility.',
     'Parking Lot 2.Another designated area for parking vehicles, providing ample space for cars and sometimes including parking structures.',
     'Tennis Court.A specialized sports surface, marked for tennis gameplay, with netting and boundary lines.',
     'Running Track.A dedicated track surface, often oval-shaped, designed for running and athletic competitions.']
    # device = 'cpu'
    #############################################################################################
    data = LIDARHS(32, mode='test')
    test_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)
    #############################################################################################
    # 定义与权重文件相匹配的模型结构
    model = UNet(8).to(device)
    net = Model_All(device).to(device)
    # 加载预训练权重文件
    # net.text_net.load_state_dict(torch.load('text_encoder.pth'))
    weights_path = 'trained_models/ddpm_mse_epochs_500_pca8.pth'
    weights_path_n = 'trained_models/clip_weightspca8.pth'
    pretrained_weights = torch.load(weights_path)
    pretrained_weights_n = torch.load(weights_path_n)
    # 将加载的权重加载到模型中
    model.load_state_dict(pretrained_weights)
    net.load_state_dict(pretrained_weights_n)

    # 将模型设置为评估模式（关闭 dropout 和 batch normalization 的训练模式）
    model.eval()
    net.eval()
    diffusion = DenoiseDiffusion(eps_model=model, n_steps=T, device=device)

    count = 0
    y_pred_test = 0
    y_test = 0
    for step, (lidar_out, HS_out, lable_out, ik_out) in enumerate(test_loader):
        lidar_out = lidar_out.type(torch.float).to(device)
        HS_out = HS_out.type(torch.float).to(device)
        hsnew = diffusion.q_sample(HS_out, torch.tensor([T - 1]).to(device))
        lable_out = lable_out.type(torch.LongTensor).to(device) - 1
        # new_hs_out = center_feature(diffusion, HS_out, T, BATCH_SIZE)
        i = ik_out[:][0]
        k = ik_out[:][1]
        with torch.no_grad():
            _, center = diffusion.ddim({0: hsnew, 1: lidar_out}, 50)
            outputs = net(center,text)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)



        if count == 0:
            y_pred_test = outputs
            y_test = lable_out.cpu()
            col = i
            nul = k
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, lable_out.cpu()))
            col = np.concatenate((col, i))
            nul = np.concatenate((nul, k))
        print(step)
    for h in range(15):print(f'{h}num is:',sum(1 for num in y_test if num==h))
    map = np.zeros((y.shape[0], y.shape[1], 3))
    map_pre = np.zeros((y.shape[0], y.shape[1], 3))
    for p in range(len(y_test)):
        item = y_test[p]
        map[col[p], nul[p]] = item_to_color(item)
        item_pre = y_pred_test[p]
        map_pre[col[p], nul[p]] = item_to_color(item_pre)
    classification_map(map, 'IP_targer_clip.png')
    classification_map(map_pre, 'IP_prediction_clippca8.png')
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    file_name = "classification_clippca8.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))



if __name__ == '__main__':
    main()