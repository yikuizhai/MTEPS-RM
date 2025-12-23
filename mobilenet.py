import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

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
