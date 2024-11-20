import torch
import torch.nn as nn
import torch.nn.functional as F
import cliptoken
from textnet import Text_Net

class TextFeatureProcessingModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextFeatureProcessingModule, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.batchnorm1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.batchnorm2 = nn.BatchNorm1d(output_dim)

    def forward(self, text_features):
        # Initial text features are transformed by the first linear layer
        output1 = F.relu(self.batchnorm1(self.linear1(text_features)))
        # The output from the first transformation is further transformed by the second linear layer
        output2 = F.relu(self.batchnorm2(self.linear2(output1)))
        # The final text features are obtained by adding the transformed output to the original features
        final_output = output2 + text_features
        return final_output


# class CrossAttentionModule(nn.Module):
#     def __init__(self,input_size):
#         super(CrossAttentionModule, self).__init__()
#         self.query_linear = nn.Linear(input_size, input_size)
#         self.key_linear = nn.Linear(input_size, input_size)
#         self.value_linear = nn.Linear(input_size, input_size)
#     def forward(self, query, key_value):
#         # 批次大小
#         batch_size = key_value.size(0)
#
#         # 调整 query 的形状，以匹配 key 和 value 的批次大小
#         # key_value 初始形状 [batch_size, 1, 512]
#         # query 初始形状 [15, 512] 需要调整为 [15, batch_size, 512]
#         query = self.query_linear(query).unsqueeze(0).expand( batch_size, -1, -1)
#         # print('q',query.shape)  #torch.Size([64, 15, 512])
#
#         # 调整 key_value 的形状，以匹配 query 的序列长度
#         # 此时 key 和 value 形状均为 [batch_size, 1, 512]
#         key = self.key_linear(key_value).unsqueeze(1)  # [batch_size, 1, 512]
#         value = self.value_linear(key_value).unsqueeze(1)  # [batch_size, 1, 512]
#         # print('k',key.shape)  #torch.Size([64, 1, 512])
#
#         d_k = key.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
#         attn_weights = F.softmax(scores, dim=-1)
#         output = torch.matmul(attn_weights, value)
#
#         return output
class CrossAttentionModule(nn.Module):
    def __init__(self,input_size):
        super(CrossAttentionModule, self).__init__()

    def forward(self, query, key_value):
        # 批次大小
        batch_size = key_value.size(0)

        # 调整 query 的形状，以匹配 key 和 value 的批次大小
        # key_value 初始形状 [batch_size, 1, 512]
        # query 初始形状 [15, 512] 需要调整为 [15, batch_size, 512]
        query = query.unsqueeze(0).expand( batch_size, -1, -1)
        # print('q',query.shape)  #torch.Size([64, 15, 512])

        # 调整 key_value 的形状，以匹配 query 的序列长度
        # 此时 key 和 value 形状均为 [batch_size, 1, 512]
        key = key_value.unsqueeze(1)  # [batch_size, 1, 512]
        value = key_value.unsqueeze(1)  # [batch_size, 1, 512]
        # print('k',key.shape)  #torch.Size([64, 1, 512])

        d_k = key.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output


class Img_Net(nn.Module):
    def __init__(self,device):
        super(Img_Net, self).__init__()
        self.projector = torch.nn.Sequential(nn.Linear(50 * 448, 512),
                                                    nn.BatchNorm1d(512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, 512)).to(device)

        self.selfAtt = nn.MultiheadAttention(embed_dim=512, num_heads=1)
        self.crossAtt = CrossAttentionModule(512)

    def forward(self,img,text):
        b = img.size(0)
        t = text.size(0)
        x = img.view(-1, 50 * 448)
        x = self.projector(x)

        crossa = self.crossAtt(text,x)
        x = x.unsqueeze(1)    #b, 1, 512

        selfa,_ = self.selfAtt(x,x,x)
        selfa = selfa.expand(b, t, 512)

        return selfa + crossa
##
class Model_All(torch.nn.Module):
    def __init__(self,device):
        super(Model_All, self).__init__()

        self.text_net = Text_Net(embed_dim=512,
                                 context_length=77,
                                 vocab_size=49408,
                                 transformer_width=512,
                                 transformer_heads=8,
                                 transformer_layers=12).to(device)

        self.text_proj = TextFeatureProcessingModule(input_dim=512,output_dim=512).to(device)

        self.img_net = Img_Net(device).to(device)
        self.device = device


    def forward(self, img, text):
        img = img.to(self.device)
        batchsize = img.size(0)

        with torch.no_grad():
            text = cliptoken.tokenize(text).to(self.device)
            Z_text = self.text_net(text)

        with torch.enable_grad():
            Z_text = self.text_proj(Z_text)
        # print(Z_text.shape)
        # print(img.shape)
        Z_vison = self.img_net(img,Z_text)

        Z_text = Z_text.unsqueeze(0).expand(batchsize, -1, -1)


        # 将两个张量相乘
        Z_vison = Z_vison / Z_vison.norm(dim=2, keepdim=True)
        Z_text = Z_text / Z_text.norm(dim=2, keepdim=True)

        Z_text = torch.transpose(Z_text, dim0=1, dim1=2)

        product_tensor = torch.matmul(Z_vison, Z_text)
        # 提取对角线元素
        product_tensor = torch.eye(Z_text.size(2)).to(self.device) * product_tensor
        diagonal_elements = torch.diagonal(product_tensor, dim1=1, dim2=2)

        return diagonal_elements

if __name__ == '__main__':
    model = Model_All('cuda')
    img = torch.randn(15,50,448)
    text = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
         'Soil', 'Water', 'Residential','Commercial', 'Road', 'Highway', 'Railway',
            'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
    out = model(img,text)
    print(out.shape)
