import numpy as np
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from clip_model import Model_All
from sklearn.metrics import accuracy_score
def shuff(classnum,chunk_size,imgdata_chunks,):
    # 类内随机打乱
    imgdata_chunks_rand = []
    for i in range(classnum):
        # 获取第一个维度的长度，即组数
        num_groups = imgdata_chunks[i].size(0)
        # 生成随机排列的索引
        permutation = torch.randperm(num_groups)
        # 根据随机排列的索引重新排列张量的维度
        shuffled_tensor = imgdata_chunks[i][permutation]
        imgdata_chunks_rand.append(shuffled_tensor)

    # 每类取一个，组合成一个训练批次
    combinations = []
    for i in range(chunk_size):
        combination = []
        # log = []
        for j in range(classnum):
            combination.append(imgdata_chunks_rand[j][i])
            # log.append(logic_chunks[j][i])
        # print(log)  #[tensor(1), tensor(2),..., tensor(14), tensor(15)]
        combinations.append(combination)
    return combinations
def setseed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def get_ground_truth(device, num_logits) -> torch.Tensor:
    labels = torch.arange(num_logits, device=device, dtype=torch.long)
    return labels
setseed(3407)
device = "cuda" if torch.cuda.is_available() else "cpu"
epoch = 100
classnum = 15
# classnum = 14

imgdata = torch.load('feature_1500')
logic = []
for i in range(0, classnum):
    logic.extend([i] * 100)
logic = torch.tensor(logic)

print('imgdata', imgdata.size())
print('logic', logic.shape)

randindex = torch.randperm(imgdata.size(0))[:64]

randomtest = imgdata[randindex]
randomlabel = logic[randindex]


chunk_size = len(imgdata) // classnum

# 将imgdata和logic分成classnum份
imgdata_chunks = imgdata.chunk(classnum)
logic_chunks = logic.chunk(classnum)


# shuff()

model = Model_All(device).to(device)
model.text_net.load_state_dict(torch.load("./text_encoder.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.0004)
# text = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
#          'Soil', 'Water', 'Residential','Commercial', 'Road', 'Highway', 'Railway',
#             'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
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

for epoc in range(epoch+1):

    combinations = shuff(classnum,chunk_size,imgdata_chunks)
    count = 0
    y_pred_test = 0
    y_test = 0

    for i, combo in enumerate(combinations):
        # print(i)
        combo = torch.stack(combo)
        # print(f"Combination {i+1}: {combo.shape}")
        output = model(combo,text)

        outputs = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = get_ground_truth(device, output.shape[0])
        if count == 0:
            y_pred_test = outputs
            y_test = labels.cpu()
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels.cpu()))

        loss = (F.cross_entropy(output, labels) + F.cross_entropy(output, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    test = model(randomtest,text)
    test = np.argmax(test.detach().cpu().numpy(),axis=1)
    print(test.shape)

    oa_t = accuracy_score(test,randomlabel)
    print("oatest",oa_t)
    # print('output',output[0])
    # print('output2',output[1])
    # print(outputs)
    # print(labels)
    print('[Epoch: %d]   [current loss: %.4f]' %(epoc + 1,loss.item()))
    oa = accuracy_score(y_test, y_pred_test)
    print(oa)
    if epoc % 10 == 0:
        torch.save(model.state_dict(), 'trained_models/clip_weightspca8.pth')

