import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpecAttentionModule, self).__init__()
        # Assuming the 'c' in the input dimension [batchsize, c, d, d] is the number of channels.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d_q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv1d_k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv1d_v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv1d_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, _ = x.shape
        # Apply avg pooling
        pooled = self.avg_pool(x).view(b, c, -1)  # Now shape is [batch, channels, 1]

        # Generate Q, K, V
        q = self.conv1d_q(pooled)
        k = self.conv1d_k(pooled)
        v = self.conv1d_v(pooled)

        # Calculate attention map
        attention_scores = torch.matmul(k,q.transpose(1, 2))  # [batch, d, d]
        attention_map = F.softmax(attention_scores, dim=-1)  # Apply softmax on the last dimension

        # Apply attention map to V
        attention_out = torch.matmul( attention_map.transpose(1, 2),v)

        # Reshape to the original size and apply 1D conv

        attention = attention_out + v
        mapout = self.sigmoid(self.conv1d_out(attention))
        mapout = mapout.unsqueeze(-1)

        # Combine with the input
        out = x + x * mapout
        return out


class SpaAttentionModule(nn.Module):
    def __init__(self, in_channels,  in_ld):
        super(SpaAttentionModule, self).__init__()
        # Convolution layers for X
        self.conv2d_x1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2d_x2 = nn.Conv2d(in_ld, in_channels, kernel_size=3, padding=1)

        # Convolution layers for Q, K, V
        self.conv2d_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2d_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2d_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Sigmoid for output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x , ld):
        # Apply convolutions on X to get X1 and X2
        x1 = self.conv2d_x1(x)
        x2 = self.conv2d_x2(ld)

        # Get Q, K, V from X1
        q = self.conv2d_q(x2)
        k = self.conv2d_k(x1)
        v = self.conv2d_v(x1)

        # Reshape Q, K, V for batch matrix multiplication
        batch_size, channels, height, width = q.size()
        q = q.view(batch_size, channels, -1)
        k = k.view(batch_size, channels, -1).permute(0, 2, 1)
        v = v.view(batch_size, channels, -1)

        # Attention mechanism
        attention_scores = torch.matmul(q, k) / (channels ** 0.5)  # Scaled Dot-Product
        attention_map = F.softmax(attention_scores, dim=-1)
        attention_out = torch.matmul(attention_map, v) + v

        # Apply maxpool and sigmoid to the output of X2
        mapout = self.sigmoid(attention_out).view(x.shape)
        # Element-wise multiplication of attention_out and x
        out = mapout * x

        # Element-wise addition of out and input X
        out = out + x

        return out


class sp_atten(nn.Module):
    def __init__(self,in_hsi, in_ld):
        super(sp_atten, self).__init__()
        self.spe = SpecAttentionModule(in_hsi)
        self.spa = SpaAttentionModule(in_hsi,in_ld)


    def forward(self,hsi,lidar):

        hsi = self.spe(hsi)
        hsi = self.spa(hsi,lidar)
        return hsi

if __name__ == '__main__':


    # Assuming the in_channels and out_channels for simplicity
    in_channels = 10
    out_channels = 64
    in_ld = 1
    attention_module = SpaAttentionModule(in_channels,in_ld)

    # Example input tensor [batchsize, c, d, d]
    example_input = torch.rand(4, in_channels, 10, 10)
    example_input2 = torch.rand(4, 1, 10, 10)

    inall = torch.concat((example_input,example_input2),dim = 1)
    s = (example_input,example_input)
    output = attention_module(s,example_input2)

    # Print output shape
    print(output.shape)
