import torch
from torch import nn
import torch.nn.functional as F

patch_size = 9

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # 1x1 conv 相当于 Linear，用于生成 θ, φ, g
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # 最终融合 1x1 conv
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)  # 稳定训练
        )
        nn.init.constant_(self.W[0].weight, 0)
        nn.init.constant_(self.W[0].bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # [B, C, H, W] → [B, C_inter, H*W]
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)
        g = self.g(x).view(batch_size, self.inter_channels, -1)

        # [B, H*W, C_inter] × [B, C_inter, H*W] → [B, H*W, H*W]
        attention = torch.bmm(theta.permute(0, 2, 1), phi)  # [B, N, N], N=H*W
        attention = F.softmax(attention, dim=-1)

        # [B, H*W, H*W] × [B, C_inter, H*W]^T → [B, H*W, C_inter]
        out = torch.bmm(g, attention.permute(0, 2, 1))  # [B, C_inter, H*W]

        out = out.view(batch_size, self.inter_channels, H, W)
        out = self.W(out)  # [B, C, H, W]

        return x + out  # 残差连接，稳定训练


class HAM(nn.Module):
    def __init__(self, in_channels, rate=3):
        super(HAM, self).__init__()
        self.in_channels = in_channels
        self.rate = rate

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # 空间注意力模块
        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            # 恢复通道数
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=5, padding=2),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            # 恢复通道数
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels)
        )

        # NonLocalBlock
        self.nlb = NonLocalBlock(in_channels)

        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, h, w = x.shape

        # 计算通道注意力
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        # 应用通道注意力
        x = x * x_channel_att

        # 两个分支处理
        x_nlb = self.nlb(x)
        x_spatial_att1 = self.spatial_attention1(x).sigmoid()
        x_spatial_att2 = self.spatial_attention2(x).sigmoid()
        x_spatial_att = x_spatial_att1 + x_spatial_att2
        x_spatial = x * x_spatial_att

        # 加权融合（保持3个通道）
        out = self.alpha * x_nlb + self.beta * x_spatial

        return out


class CAAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(CAAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self._init_sobel_kernels()

    def _init_sobel_kernels(self):
        # 创建Sobel滤波器核
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        # 并注册为缓冲区
        self.register_buffer('sobel_x', sobel_x.view(1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 3, 3))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "contour_simam"

    def _compute_contour_strength(self, x):
        """
        计算输入特征的轮廓强度
        """
        b, c, h, w = x.size()

        sobel_x_weight = self.sobel_x.expand(c, 1, 3, 3)
        sobel_y_weight = self.sobel_y.expand(c, 1, 3, 3)

        # 应用卷积，groups=c 确保每个通道独立计算
        grad_x = F.conv2d(x, sobel_x_weight, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y_weight, padding=1, groups=c)

        # 计算梯度幅度（轮廓强度）
        contour_strength = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.e_lambda)

        return contour_strength

    def forward(self, x):
        b, c, h, w = x.size()

        # 计算多方向梯度强度作为轮廓度量
        contour_strength = self._compute_contour_strength(x)
        # 归一化轮廓强度
        contour_norm = contour_strength / (4 * (contour_strength.mean(dim=[2, 3], keepdim=True) + self.e_lambda)) + 0.5

        # 返回经过轮廓增强注意力加权的特征
        return x * self.activation(contour_norm)


class CSFNet(nn.Module):
  def __init__(self):
    super(CSFNet, self).__init__()
    self.caam = CAAM()
    self.ham = HAM(3)
    self.linear1=nn.Linear(patch_size * patch_size * 6, 50)
    self.linear2=nn.Linear(50, 20)
    self.linear3=nn.Linear(20, 2)

  def forward(self, img):
    in_x = img.reshape(img.shape[0],-1)
    hamOut = self.ham(img)
    hamout = hamOut.reshape(hamOut.shape[0], 1, -1)
    camOut = self.caam(img)
    camout = camOut.reshape(camOut.shape[0],1,-1)
    catOut = torch.cat((hamOut, camOut), 1)
    catout = catOut.reshape(catOut.shape[0],1,-1)
    x = catOut

    fuseout = x.reshape(x.shape[0],1,-1)
    out1 = x.reshape(x.size(0), -1)
    out1 = self.linear1(out1)
    out1 = self.linear2(out1)
    out1 = self.linear3(out1)
    return in_x, hamout, camout, catout, fuseout, out1





