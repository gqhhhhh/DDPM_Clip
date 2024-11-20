import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, NUM_CLASS = 15):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(50*448, 50*10)
        self.fc2 = nn.Linear(50*10, 50*50)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.2)
        self.numcls = NUM_CLASS
        self.fc3 = nn.Linear(448*50, self.numcls)

    def forward(self, x):
        # 将输入展平为一维向量
        xt = x.view(-1, 50*448)
        # 第一个全连接层
        xt = F.relu(self.fc1(xt))
        xt = self.dropout(xt)
        # 第二个全连接层
        xt = F.softmax(self.fc2(xt),dim=1)
        xt = xt.view(-1,50,50)

        x = torch.einsum('bij,bjk->bik', xt, x)
        x = x.view(-1, 50 * 448)
        x = F.softmax(self.fc3(x),dim=1)

        return x

if __name__ == '__main__':

    # 创建模型实例
    model = SimpleModel()
    # 打印模型结构
    # print(model)
    a = torch.rand(64, 50, 448)
    b = model(a)
    print(b.shape)
    print(b[1])
