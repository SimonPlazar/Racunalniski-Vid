import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic building blocks
# -------------------------

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Down block:
        Conv -> BN -> ReLU
        Conv -> BN -> ReLU
        MaxPool2d(2,2)
    """
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, k)
        self.conv2 = ConvBlock(out_ch, out_ch, k)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x          # Used for skip-connection
        x = self.pool(x)
        return x, before_pool


# -------------------------
# FlowNetSimple network
# -------------------------

class FlowNetSimple(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder -----
        self.down1 = DownBlock(6, 16, 7)    # input: 2 RGB frames -> 6 channels
        self.down2 = DownBlock(16, 32, 5)
        self.down3 = DownBlock(32, 64, 3)

        # Bottleneck
        self.b1 = ConvBlock(64, 128, 3)
        self.b2 = ConvBlock(128, 128, 3)

        # ----- Decoder -----
        # Upsample + concat with skip connections
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)

        # ----- Multi-scale flow predictions -----
        self.predict3 = nn.Conv2d(128, 2, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1 = nn.Conv2d(32, 2, kernel_size=1)

        # Upsample predicted flow to next level
        self.up_flow3 = nn.ConvTranspose2d(2, 16, kernel_size=2, stride=2)
        self.up_flow2 = nn.ConvTranspose2d(2, 16, kernel_size=2, stride=2)

    def forward(self, x):
        # ----- Encoder -----
        x1, skip1 = self.down1(x)   # (B,16,H/2)
        x2, skip2 = self.down2(x1)  # (B,32,H/4)
        x3, skip3 = self.down3(x2)  # (B,64,H/8)

        # bottleneck
        xb = self.b1(x3)
        xb = self.b2(xb)

        # ----- Decoder -----

        # Level 3 (lowest resolution)
        up3 = self.up3(xb)
        concat3 = torch.cat([up3, skip3], dim=1)   # (64 + 64 = 128 channels)
        flow3 = self.predict3(concat3)             # 1st flow output
        up_flow3 = self.up_flow3(flow3)

        # Level 2
        up2 = self.up2(concat3)
        concat2 = torch.cat([up2, skip2, up_flow3], dim=1)  # (32 + 32 + 2)
        flow2 = self.predict2(concat2)
        up_flow2 = self.up_flow2(flow2)

        # Level 1 (full resolution)
        up1 = self.up1(concat2)
        concat1 = torch.cat([up1, skip1, up_flow2], dim=1)   # (16 + 16 + 2)
        flow1 = self.predict1(concat1)                      # final prediction

        return flow1, flow2, flow3

class MultiScaleEPE(nn.Module):
    """
    Multi-scale Endpoint Error loss
    EPE = Average Euclidean distance between predicted and ground truth flow
    """
    def __init__(self, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.weights = weights

    def forward(self, flow_pred, flow_gt):
        """
        Args:
            flow_pred: tuple of (flow1, flow2, flow3) at different scales
            flow_gt: ground truth flow at full resolution (B, 2, H, W)
        """
        flow1, flow2, flow3 = flow_pred

        # Downsample ground truth to match prediction scales
        h, w = flow_gt.shape[2:]

        flow_gt_2 = F.interpolate(flow_gt, size=(h//2, w//2), mode='bilinear', align_corners=False) * 0.5
        flow_gt_3 = F.interpolate(flow_gt, size=(h//4, w//4), mode='bilinear', align_corners=False) * 0.25

        # Compute EPE at each scale
        epe1 = torch.norm(flow1 - flow_gt, p=2, dim=1).mean()
        epe2 = torch.norm(flow2 - flow_gt_2, p=2, dim=1).mean()
        epe3 = torch.norm(flow3 - flow_gt_3, p=2, dim=1).mean()

        # Weighted sum
        total_loss = (self.weights[0] * epe1 +
                     self.weights[1] * epe2 +
                     self.weights[2] * epe3)

        return total_loss, epe1, epe2, epe3