import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT


class prelu(nn.Module):
    def __init__(self):
        super(prelu, self).__init__()
        self.p=torch.nn.PReLU()


    def forward(self, x):
        x=self.p(x)
        return x

class tsc(nn.Module):
    def __init__(self, channels):
        super(tsc, self).__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            prelu(),
            nn.Conv2d(channels//2, channels, 3, 1, 1, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out

class cpc(nn.Module):
    def __init__(self, channels):
        super(cpc, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1, bias=True),
            prelu(),
            nn.Conv2d(channels//2, channels, 3, 1, 1, bias=True),
        )

    def __call__(self, x):
        out = self.body(x)
        return out

class Conv2dSWL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_radius=2, bias=True):
        super(Conv2dSWL, self).__init__()

        kernel_size_h = 2 * kernel_radius - 1
        self.padding = kernel_radius - 1

        self.convL = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size_h, kernel_radius),
            padding=self.padding,
            bias=bias)

    def forward(self, input):
        out_L = self.convL(input)
        return out_L[:, :, :, :-self.padding]

class Conv2dSWR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_radius=2, bias=True):
        super(Conv2dSWR, self).__init__()

        kernel_size_h = 2 * kernel_radius - 1
        self.padding = kernel_radius - 1

        self.convR = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size_h, kernel_radius),
            padding=self.padding,
            bias=bias)

    def forward(self, input):
        out_R = self.convR(input)
        return out_R[:, :, :, self.padding:]

class Conv2dSWU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_radius=2, bias=True):
        super(Conv2dSWU, self).__init__()

        kernel_size_h = 2 * kernel_radius - 1
        self.padding = kernel_radius - 1

        self.convU = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_radius, kernel_size_h),
            padding=self.padding,
            bias=bias)

    def forward(self, input):
        out_U = self.convU(input)
        return out_U[:, :, :-self.padding, :]

class Conv2dSWD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_radius=2, bias=True):
        super(Conv2dSWD, self).__init__()

        kernel_size_h = 2 * kernel_radius - 1
        self.padding = kernel_radius - 1

        self.convD = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_radius, kernel_size_h),
            padding=self.padding,
            bias=bias)

    def forward(self, input):
        out_D = self.convD(input)
        return out_D[:, :, self.padding:, :]

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation=None, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, input_size, kernel_size, stride, padding, groups=input_size,
                                               bias=bias)
        self.deconv_ = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1, padding=0,
                                 groups=1)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv_(self.deconv(x)))
        else:
            out = self.deconv_(self.deconv(x))

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class HSF(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, norm=None):
        super(HSF, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.body1 = cpc(in_channels)
        self.c1 = Conv2dSWL(in_channels, in_channels, 2)
        self.c2 = Conv2dSWR(self.remaining_channels, in_channels, 2)
        self.c3 = Conv2dSWU(self.remaining_channels*2, in_channels, 2)
        self.c4 = Conv2dSWD(self.remaining_channels*3, in_channels, 2)
        self.act = prelu()
        self.c5 = nn.Conv2d(in_channels, self.distilled_channels, 3, padding=1, bias=True)
        self.c6 = nn.Conv2d(self.distilled_channels * 4, in_channels, 3, padding=1, bias=True)


    def forward(self, input):
        input1 = self.body1(input)
        out_c1 = self.act(self.c1(input1))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        concat1 = torch.cat((remaining_c1, remaining_c2), 1)
        out_c3 = self.act(self.c3(concat1))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        concat2 = torch.cat((remaining_c3, concat1), 1)
        out_c4 = self.act(self.c4(concat2))

        distilled_c4 = self.act(self.c5(out_c4))
        out2 = torch.cat((distilled_c1, distilled_c2, distilled_c3, distilled_c4), dim=1)

        out_fused = self.act(self.c6(out2)) + input
        return out_fused

class UDC(torch.nn.Module):
    def __init__(self, in_filter, channel, kernel_size=8, stride=4, padding=2, activation='prelu',norm=None):
        super(UDC, self).__init__()
        self.up_conv1 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation=None, norm=None)

        self.down1 = ConvBlock(in_filter, channel, kernel_size, stride, padding, activation, norm=None)
        self.down2 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.down3 = ConvBlock(channel, channel, kernel_size, stride, padding, activation=None, norm=None)

    def forward(self, x):
        x0 = self.up_conv1(x)
        x1 = self.up_conv2(x0)
        x2 = self.up_conv3(x1-x)
        out_ha =x0 + x2

        y0 = self.down1(out_ha)
        y1 = self.down2(y0)
        y2 = self.down3(y1-out_ha)
        out_la = y0 + y2

        return  out_la

class DUC(torch.nn.Module):
    def __init__(self, in_filter, channel, kernel_size=8, stride=4, padding=2,  activation='prelu',
                 norm=None):
        super(DUC, self).__init__()
        self.down1 = ConvBlock(in_filter, channel, kernel_size, stride, padding, activation, norm=None)
        self.down2 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.down3 = ConvBlock(channel, channel, kernel_size, stride, padding, activation=None, norm=None)

        self.up_conv1 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(channel, channel, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(channel, channel, kernel_size, stride, padding, activation=None, norm=None)

    def forward(self, x):
        y0 = self.down1(x)
        y1 = self.down2(y0)
        y2 = self.down3(y1 - x)

        out_la = y0 + y2

        x0 = self.up_conv1(out_la)
        x1 = self.up_conv2(x0)
        x2 = self.up_conv3(x1 - out_la)
        out_ha = x0 + x2

        return out_ha

class ChannelAttention(nn.Module):
    def __init__(self, channel,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel//ratio, 1, bias=False)
        self.act1 = prelu()
        self.fc2 = nn.Conv2d(channel//ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,  kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in(3, 7)
        padding = 3 if kernel_size ==7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        x = torch.cat([avg_out, max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MCF(nn.Module):
    def __init__(self, channel):
        super(MCF, self).__init__()
        self.channel = channel
        self.FB_1 =UDC(channel,channel)
        self.FB_2 =DUC(channel,channel)

        self.CA1 = ChannelAttention(channel,ratio=16)
        self.SA1 = SpatialAttention(kernel_size=7)
        self.CA2 = ChannelAttention(channel, ratio=16)
        self.SA2 = SpatialAttention(kernel_size=7)

        self.SFLayers = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 1, padding=0, bias=True),
            prelu(),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            prelu(),
            nn.ConvTranspose2d(channel, channel, kernel_size=6, stride=2, padding=2, output_padding=0),
            prelu(),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
        )

        self.conv = ConvBlock(2 * channel, channel, 1, 1, 0, activation='prelu', norm=None)
    def forward(self, x1, x2):
        x_1 = self.FB_1(x1)
        x__1= self.FB_2(x1)
        x_2 = self.FB_1(x2)
        x__2= self.FB_2(x2)

        y_1 = self.CA1(x_1)*x_1#[8,32,1,1]######F

        y_2 = self.CA2(x_2)*x_2#[8,32,1,1]######F

        z_1 = self.SA1(y_1)#[8,1,16,16]######W

        z_2 = self.SA2(y_2)#[8,1,16,16]######W


        y__1 = self.CA1(x__1)*x__1#[8,32,1,1]######F

        y__2 = self.CA2(x__2)*x__2#[8,32,1,1]######F

        z__1 = self.SA1(y__1)#[8,1,16,16]######W

        z__2 = self.SA2(y__2)#[8,1,16,16]######W


        z1 = y_1 * z_2#[8, 32, 16, 16]######F######W
        z2 = y_2 * z_1#[8, 32, 16, 16]######F######W


        Z1 = y__1 * z__2#[8, 32, 16, 16]
        Z2 = y__2 * z__1#[8, 32, 16, 16]


        c0 = self.SFLayers(torch.cat([z1, z2], 1))#[8, 32, 32, 32]
        c1 = self.SFLayers(torch.cat([Z1, Z2], 1))#[8, 32, 32, 32]

        c = self.conv(torch.cat([c0,c1],1))

        return c

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, ):
        super(Net, self).__init__()
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat_color0 = ConvBlock(3, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        

        self.fe1_Conv1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.fe1_Conv2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.fe1_Conv3 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.fe2_Conv1 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.fe2_Conv2 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.fe2_Conv3 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.down_0 = ConvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.down_1 = ConvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.down_2 = ConvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.down_3 = ConvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)
        self.down_4 = ConvBlock(base_filter, base_filter, 6, 2, 2, activation='prelu', norm=None)

        self.Def1 = HSF(base_filter ,  norm=None)
        self.C1ef1 = HSF(base_filter,  norm=None)
        self.C2ef1 = HSF(base_filter,  norm=None)
        self.C3ef1 = HSF(base_filter,  norm=None)
        self.C4ef1 = HSF(base_filter,  norm=None)

        self.tsc0 = tsc(base_filter)
        self.tsc1 = tsc(base_filter)
        self.tsc2 = tsc(base_filter)
        self.tsc3 = tsc(base_filter)

        self.sf0 = MCF(base_filter )
        self.sf1 = MCF(base_filter)
        self.sf2 = MCF(base_filter)
        self.sf3 = MCF(base_filter)

        self.up_conv = DeconvBlock(base_filter, base_filter, kernel_size=12, stride=8, padding=2, norm=None)
        self.Out_Conv = ConvBlock(base_filter, 1, 3, 1, 1, norm=None)

    def forward(self, rgb, depth):
        x0 = self.feat0(depth)
        x1 = self.feat1(x0)

        c0 = self.feat_color0(rgb)
        c1 = self.feat_color1(c0)

        D_image0 = self.down_0(x1)
        D0 = self.Def1(D_image0)

        ######## 
        C_image0 = self.down_1(c1)
        C0 = self.C1ef1(C_image0)

        C_image1 = self.down_2(C0)
        C1 = self.C2ef1(C_image1)

        C_image2 = self.down_3(C1)
        C2 = self.C3ef1(C_image2)

        C_image3 = self.down_4(C2)
        C3 = self.C4ef1(C_image3)


        F0 = self.sf0(D0, C3)  # 32-(32*32)

        D_up0 = self.tsc0(D0)
        F_0 = F0 + D_up0 #32-(32*32)

        C_2 = self.fe2_Conv1(self.fe1_Conv1(C2))
        F1 = self.sf1(F_0, C_2)  # 32-(64*64)

        F_X0 = self.tsc1(F_0)
        F_1 = F1 + F_X0 #32-(64*64)

        C_1 = self.fe2_Conv2(self.fe1_Conv2(C1))
        F2 = self.sf2(F_1, C_1)  # 32-(128*128)


        F_X1 = self.tsc2(F_1)
        F_2 = F2 + F_X1 #32-(128*128)

        C_0 = self.fe2_Conv3(self.fe1_Conv3(C0))
        F3 = self.sf3(F_2, C_0)  # 32-(256*256)

        F_X2 = self.tsc3(F_2)
        X1 = self.up_conv(x1)
        F_3 = F3 + F_X2 + X1
        F = self.Out_Conv(F_3)
        return F
