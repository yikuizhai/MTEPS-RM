import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

# ======= 上采样模块 =======
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

# ======= 变化检测 MobileNetV2-UNet =======
class ChangeDetectionMobileNetV2UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ChangeDetectionMobileNetV2UNet, self).__init__()

        # 裁掉最后 1280 通道层（features[-1]）
        backbone = mobilenet_v2(pretrained=True).features[:-1]

        # MobileNetV2主要特征层通道数 (去掉1280层)
        self.enc_channels = [16, 24, 32, 96, 320]  # 对应特征输出

        # 分别创建两个分支
        self.encoder1 = backbone
        self.encoder2 = mobilenet_v2(pretrained=True).features[:-1]

        # Decoder 通道数（拼接后通道数*2）
        self.up5 = UpBlock(self.enc_channels[-1] * 2, 320)           # 320*2 -> 320
        self.up4 = UpBlock(320 + self.enc_channels[-2] * 2, 160)     # 320+96*2 -> 160
        self.up3 = UpBlock(160 + self.enc_channels[-3] * 2, 96)      # 160+32*2 -> 96
        self.up2 = UpBlock(96 + self.enc_channels[-4] * 2, 64)       # 96+24*2 -> 64
        self.up1 = UpBlock(64 + self.enc_channels[-5] * 2, 32)       # 64+16*2 -> 32

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, t1, t2):
        # ============ 编码器部分 ============
        feats1 = []
        feats2 = []
        x1 = t1
        x2 = t2
        for i, layer in enumerate(self.encoder1):
            x1 = layer(x1)
            x2 = self.encoder2[i](x2)
            # 选取下采样后的关键层输出
            if i in [1, 3, 6, 13, 17]:  # 对应 [16,24,32,96,320]
                feats1.append(x1)
                feats2.append(x2)

        # feats1/feats2: 5个尺度, 通道=[16,24,32,96,320]
        feats1 = feats1[::-1]  # [320,96,32,24,16]
        feats2 = feats2[::-1]

        # ============ 解码器部分 ============
        d5 = torch.cat([feats1[0], feats2[0]], dim=1)  # 320*2=640
        d5 = self.up5(d5)  # -> 320

        d4 = torch.cat([d5, feats1[1], feats2[1]], dim=1)  # 320+96*2=512
        d4 = self.up4(d4)  # -> 160

        d3 = torch.cat([d4, feats1[2], feats2[2]], dim=1)  # 160+32*2=224
        d3 = self.up3(d3)  # -> 96

        d2 = torch.cat([d3, feats1[3], feats2[3]], dim=1)  # 96+24*2=144
        d2 = self.up2(d2)  # -> 64

        d1 = torch.cat([d2, feats1[4], feats2[4]], dim=1)  # 64+16*2=96
        d1 = self.up1(d1)  # -> 32

        out = self.final_conv(d1)
        return out

# ===================== 测试 FLOPs 和 参数量 =====================
if __name__ == "__main__":
    from thop import profile

    model = ChangeDetectionMobileNetV2UNet(num_classes=2)
    dummy_input1 = torch.randn(1, 3, 256, 256)
    dummy_input2 = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(dummy_input1, dummy_input2))
    print(f"Params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}G")


def OA(pre_classes, gt_classes):
    return torch.sum((pre_classes) == (gt_classes)).float()/len(pre_classes)

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_Encoder(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, nl=32):
        super(UNet_Encoder, self).__init__()

        n1 = nl
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x):
        es = {}
        
        e1 = self.Conv1(x)
        es['e1']=e1
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        es['e2']=e2

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        es['e3']=e3

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        es['e4']=e4

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        es['e5']=e5

        return es

class UNet_Decoder_I(nn.Module):
    def __init__(self, out_ch=2, nl = 64):
        super(UNet_Decoder_I, self).__init__()

        n1 = nl
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, e):
        e5 = e['e5']
        e4 = e['e4']
        e3 = e['e3']
        e2 = e['e2']
        e1 = e['e1']
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv(d2)

        return out


class UNet_Decoder_S(nn.Module):
    def __init__(self, out_ch=2, nl = 128):
        super(UNet_Decoder_S, self).__init__()

        n1 = nl
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

       # self.active = torch.nn.Sigmoid()

    def forward(self, t1_e, t2_e):
        e5 = torch.cat((t1_e['e5'], t2_e['e5']),dim=1)
        e4 = torch.cat((t1_e['e4'], t2_e['e4']),dim=1)
        e3 = torch.cat((t1_e['e3'], t2_e['e3']),dim=1)
        e2 = torch.cat((t1_e['e2'], t2_e['e2']),dim=1)
        e1 = torch.cat((t1_e['e1'], t2_e['e1']),dim=1)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv(d2)

        return out

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, nl=16):
        super(UNet, self).__init__()
        self.encoder = UNet_Encoder(in_ch,nl)
        self.decoder_s = UNet_Decoder_S(out_ch,nl=nl*2)

    def forward(self, t1_i,t2_i):
        t1_f = self.encoder(t1_i)
        t2_f = self.encoder(t2_i)
        s = self.decoder_s(t1_f,t2_f)
        return s
    
class UNet_seg(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, nl=16):
        super(UNet_seg, self).__init__()
        self.encoder = UNet_Encoder(in_ch,nl)
        self.decoder = UNet_Decoder_I(out_ch,nl)

    def forward(self, i):
        f = self.encoder(i)
        s = self.decoder(f)
        return s