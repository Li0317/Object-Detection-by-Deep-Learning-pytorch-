import torch
from torch import nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding= 0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding)
    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        #inplace参数对计算结果不会有影响。利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。
        #但是会对原变量覆盖，只要不带来错误就用
        return x

class inceptionv1(nn.Module):
    def __init__(self, input_dim, hidden_1_1, hidden_2_1, hidden_2_3, hidden_3_1, output_3_5, output_4_1):
        super(inceptionv1,self).__init__()
        self.branch1x1 = BasicConv2d(input_dim, hidden_1_1, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_dim, hidden_2_1, 1),
            BasicConv2d(hidden_2_1, hidden_2_3, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_dim, hidden_3_1, 1),
            BasicConv2d(hidden_3_1, output_3_5, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride= 1, padding= 1),
            BasicConv2d(input_dim, output_4_1, 1)
        )

    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(x)
        x3 = self.branch5x5(x)
        x4 = self.branch_pool(x)
        output = torch.cat((x1, x2, x3, x4), dim=1)
        return output

inceptionv1 = inceptionv1(3, 64, 32, 64, 64, 96, 32)
input = torch.randn(1, 3, 256, 256)
output = inceptionv1(input)

print(output.shape)