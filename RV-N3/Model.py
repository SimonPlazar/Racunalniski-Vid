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
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
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
        before_pool = x  # Used for skip-connection
        x = self.pool(x)
        return x, before_pool


# -------------------------
# FlowNetSimple network
# -------------------------

class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder (4 nivoji, več kanalov) -----
        self.down1 = DownBlock(6, 64, 7)  # 6 -> 64 (namesto 16)
        self.down2 = DownBlock(64, 128, 5)  # 64 -> 128 (namesto 32)
        self.down3 = DownBlock(128, 256, 3)  # 128 -> 256 (namesto 64)
        self.down4 = DownBlock(256, 512, 3)  # Nov nivo

        # Bottleneck
        self.b1 = ConvBlock(512, 512, 3)
        self.b2 = ConvBlock(512, 512, 3)

        # ----- Decoder -----
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2)  # 768→256
        self.up2 = nn.ConvTranspose2d(514, 128, kernel_size=2, stride=2)  # 256+256+2=514
        self.up1 = nn.ConvTranspose2d(258, 64, kernel_size=2, stride=2)  # 128+128+2=258

        # ----- Multi-scale predictions -----
        self.predict4 = nn.Conv2d(768, 2, kernel_size=1)  # 256+512=768
        self.predict3 = nn.Conv2d(514, 2, kernel_size=1)  # 256+256+2=514
        self.predict2 = nn.Conv2d(258, 2, kernel_size=1)  # 128+128+2=258
        self.predict1 = nn.Conv2d(130, 2, kernel_size=1)  # 64+64+2=130

    def forward(self, x):
        # Encoder
        x1, skip1 = self.down1(x)  # /2
        x2, skip2 = self.down2(x1)  # /4
        x3, skip3 = self.down3(x2)  # /8
        x4, skip4 = self.down4(x3)  # /16 (nov)

        # Bottleneck
        xb = self.b1(x4)
        xb = self.b2(xb)

        # Decoder
        # Level 4 (1/16)
        up4 = self.up4(xb)
        concat4 = torch.cat([up4, skip4], dim=1)
        flow4 = self.predict4(concat4)

        # Level 3 (1/8)
        up3 = self.up3(concat4)
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=False)
        concat3 = torch.cat([up3, skip3, flow4_up], dim=1)
        flow3 = self.predict3(concat3)

        # Level 2 (1/4)
        up2 = self.up2(concat3)
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=False)
        concat2 = torch.cat([up2, skip2, flow3_up], dim=1)
        flow2 = self.predict2(concat2)

        # Level 1 (full res)
        up1 = self.up1(concat2)
        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=False)
        concat1 = torch.cat([up1, skip1, flow2_up], dim=1)
        flow1 = self.predict1(concat1)

        return flow1, flow2, flow3, flow4


class MultiScaleEPE(nn.Module):
    """
    Multi-scale Endpoint Error loss
    EPE = Average Euclidean distance between predicted and ground truth flow
    """

    def __init__(self, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.weights = weights

    def forward(self, flow_pred, flow_gt):
        """
        Args:
            flow_pred: tuple of (flow1, flow2, flow3) at different scales
            flow_gt: ground truth flow at full resolution (B, 2, H, W)
        """
        flow1, flow2, flow3, flow4 = flow_pred
        h, w = flow_gt.shape[2:]

        flow_gt_2 = F.interpolate(flow_gt, size=(h // 2, w // 2), mode='bilinear') * 0.5
        flow_gt_3 = F.interpolate(flow_gt, size=(h // 4, w // 4), mode='bilinear') * 0.25
        flow_gt_4 = F.interpolate(flow_gt, size=(h // 8, w // 8), mode='bilinear') * 0.125

        epe1 = torch.norm(flow1 - flow_gt, p=2, dim=1).mean()
        epe2 = torch.norm(flow2 - flow_gt_2, p=2, dim=1).mean()
        epe3 = torch.norm(flow3 - flow_gt_3, p=2, dim=1).mean()
        epe4 = torch.norm(flow4 - flow_gt_4, p=2, dim=1).mean()

        total_loss = (self.weights[0] * epe1 + self.weights[1] * epe2 +
                      self.weights[2] * epe3 + self.weights[3] * epe4)
        return total_loss, epe1, epe2, epe3, epe4


# Dataset
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import IO


class FlyingChairsDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform

        split_file = self.root / "FlyingChairs_train_val.txt"
        split_ids = np.loadtxt(split_file, dtype=np.int32)

        if split == "train":
            self.indices = np.where(split_ids == 1)[0] + 1
        elif split == "val":
            self.indices = np.where(split_ids == 2)[0] + 1
        else:
            self.indices = np.arange(1, len(split_ids) + 1)

        self.data_dir = self.root / "data"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n = int(self.indices[idx])
        img1_path = str(self.data_dir / f"{n:05d}_img1.ppm")
        img2_path = str(self.data_dir / f"{n:05d}_img2.ppm")
        flow_path = str(self.data_dir / f"{n:05d}_flow.flo")

        # Read as numpy arrays
        img1 = IO.readImage(img1_path)  # H×W×3 uint8
        img2 = IO.readImage(img2_path)
        flow = IO.read(flow_path)  # H×W×2 float32

        # Get original size before any transforms
        orig_h, orig_w = img1.shape[:2]

        # Apply transform if exists (expects PIL or numpy)
        if self.transform:
            # Convert to PIL for torchvision transforms
            from PIL import Image
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

            img1 = self.transform(img1)  # now it's a tensor 3×256×256
            img2 = self.transform(img2)

            # Get new size from transformed tensor
            _, new_h, new_w = img1.shape

            # Resize and scale flow
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # 2×H×W
            flow = F.interpolate(flow.unsqueeze(0), size=(new_h, new_w),
                                 mode='bilinear', align_corners=False).squeeze(0)

            # Scale flow values proportionally
            scale_h = new_h / orig_h
            scale_w = new_w / orig_w
            flow[0] *= scale_w  # u component
            flow[1] *= scale_h  # v component
        else:
            # No transform: just convert to tensors
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # Stack images → 6×H×W
        img_pair = torch.cat([img1, img2], dim=0)

        return img_pair, flow
