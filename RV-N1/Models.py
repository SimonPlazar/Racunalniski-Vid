import torch.nn as nn


# ============================================================
# RESNET
# ============================================================

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1,
                 dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        out_channels = out_channels or in_channels  # če ni določeno, ohrani enako št. kanalov

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout(out)
        out += residual

        out = self.relu(out)
        return out


class ResNetBody(nn.Module):
    def __init__(self, in_channels=2, dropout_rate=0.1):
        super(ResNetBody, self).__init__()

        # ----- 1. stopnja -----
        self.layer1 = nn.Sequential(
            ResNetBlock(in_channels, 64, dropout_rate=dropout_rate),
            ResNetBlock(64, 64, dropout_rate=dropout_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        )

        # ----- 2. stopnja -----
        self.layer2 = nn.Sequential(
            ResNetBlock(64, 64, dropout_rate=dropout_rate),
            ResNetBlock(64, 64, dropout_rate=dropout_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )

        # ----- 3. stopnja -----
        self.layer3 = nn.Sequential(
            ResNetBlock(64, 128, dropout_rate=dropout_rate),
            ResNetBlock(128, 128, dropout_rate=dropout_rate),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # ----- 4. stopnja -----
        self.layer4 = nn.Sequential(
            ResNetBlock(128, 128, dropout_rate=dropout_rate),
            ResNetBlock(128, 128, dropout_rate=dropout_rate),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            # zadnji max pool ni potreben, ohranimo 8x8
        )

        # ----- Polno povezan sloj -----
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, 512)

    def forward(self, x):  # Nx2x64x64
        x = self.layer1(x)  # Nx64x32x32
        x = self.layer2(x)  # Nx64x16x16
        x = self.layer3(x)  # Nx128x8x8
        x = self.layer4(x)  # Nx128x8x8
        x = self.flatten(x)  # Nx8192
        x = self.fc(x)  # Nx512
        return x


# ============================================================
# REGRESSION MODEL
# ============================================================

class RegressionHead(nn.Module):
    def __init__(self, in_features=512, out_features=8):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):  # Nx512
        return self.fc(x)  # Nx8


class HomographyRegressor(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(HomographyRegressor, self).__init__()
        self.body = ResNetBody(in_channels=2, dropout_rate=dropout_rate)
        self.head = RegressionHead(in_features=512, out_features=8)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_features=512, num_classes=21, class_dim=8):
        super(ClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.fc = nn.Linear(in_features, class_dim * num_classes)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x):  # Nx512
        x = self.fc(x)  # Nx168
        x = x.view(-1, self.class_dim, self.num_classes)  # Nx8x21
        # x = self.softmax(x)  # Nx8x21
        return x


class HomographyClassifier(nn.Module):
    def __init__(self, num_classes=21, class_dim=8, dropout_rate=0.1):
        super(HomographyClassifier, self).__init__()
        self.body = ResNetBody(in_channels=2, dropout_rate=dropout_rate)
        self.head = ClassificationHead(in_features=512,
                                       num_classes=num_classes,
                                       class_dim=class_dim)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


# ============================================================
# DATASETS
# ============================================================

from torch.utils.data import Dataset
from torch import from_numpy
import random

from Generator import generate_pair


class HomographyPairDataset(Dataset):
    def __init__(self, images, samples_per_epoch, window_size=64, margin=16, disp_range=(-16, 16)):
        self.images = images if isinstance(images, list) else [images]
        self.samples_per_epoch = samples_per_epoch
        self.window_size = window_size
        self.margin = margin
        self.disp_range = disp_range

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Generate a random pair from a random image
        img = random.choice(self.images)
        pair, offsets, _, _ = generate_pair(
            img,
            window_size=self.window_size,
            margin=self.margin,
            disp_range=self.disp_range
        )

        # Convert to tensors
        pair = from_numpy(pair).permute(2, 0, 1).float()  # 2x64x64
        offsets = from_numpy(offsets.flatten()).float()  # 8

        return pair, offsets


class FixedSrcRandomDispDataset(Dataset):
    def __init__(self, image, src_corners, samples_per_epoch, disp_range=(-16, 16)):
        self.image = image
        self.src_corners = np.array(src_corners, dtype=np.float32)  # (4, 2)
        self.samples_per_epoch = samples_per_epoch
        self.disp_range = disp_range

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Generate random displacements
        displacements = np.random.uniform(
            self.disp_range[0],
            self.disp_range[1],
            size=(4, 2)
        ).astype(np.float32)  # (4, 2)

        # dst_corners = src_corners + displacements
        dst_corners = self.src_corners + displacements

        # Compute homography
        H, _ = cv2.findHomography(self.src_corners, dst_corners)

        # Warp image
        warped = cv2.warpPerspective(self.image, H, (self.image.shape[1], self.image.shape[0]))

        # Extract patches
        x, y = int(self.src_corners[0, 0]), int(self.src_corners[0, 1])
        patch_size = 64

        patch_src = self.image[y:y + patch_size, x:x + patch_size]
        patch_dst = warped[y:y + patch_size, x:x + patch_size]

        # Stack patches
        pair = np.stack([patch_src, patch_dst], axis=0)  # (2, 64, 64)

        # Flatten displacements to (8,)
        offsets = displacements.flatten()  # [dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3]

        return torch.from_numpy(pair).float() / 255.0, torch.from_numpy(offsets).float()


# ============================================================
# TRAIN LOOPS
# ============================================================

import re
import os


def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, max_checkpoints=4):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

    # Keep only last N checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
    checkpoints = sorted(checkpoints, key=extract_epoch)
    while len(checkpoints) > max_checkpoints:
        old_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        os.remove(old_ckpt)
        checkpoints.pop(0)


def load_latest_checkpoint(checkpoint_dir, model, optimizer, device):
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
    checkpoints = sorted(checkpoints, key=extract_epoch)

    if checkpoints:
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"✅ Resuming from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        return start_epoch
    else:
        print("🚀 Starting training from scratch.")
        return 0


# ============================================================
# METRICS AND UTILITIES
# ============================================================

import torch


def offsets_to_class_indices(offsets, num_classes=21, disp_range=(-16, 16)):
    """Quantize continuous offsets (B,8) -> class indices (B,8) using nearest-bin rounding."""
    min_disp, max_disp = disp_range
    bin_size = (max_disp - min_disp) / (num_classes - 1)  # float
    idx = torch.round((offsets - min_disp) / bin_size).long()
    return torch.clamp(idx, 0, num_classes - 1)


def classes_to_offsets(logits, disp_range=(-16, 16), soft=False):
    """
    logits: (B, 8, 21)
    Returns:
        (B, 8) offsets in pixels
    """
    B, P, C = logits.shape

    # Create discrete displacement bins
    bins = torch.linspace(disp_range[0], disp_range[1], C, device=logits.device)

    if soft:
        probs = torch.softmax(logits, dim=-1)  # (B,8,21)
        offsets = (probs * bins).sum(dim=-1)  # weighted mean → (B,8)
    else:
        class_idx = logits.argmax(dim=-1)  # (B,8)
        offsets = bins[class_idx]  # (B,8)

    return offsets


def classification_loss(ce, logits, gt_offsets, disp_range=(-16, 16), num_classes=21):
    B = logits.shape[0]

    target = offsets_to_class_indices(
        gt_offsets,
        num_classes=num_classes,
        disp_range=disp_range
    )  # (B,8)

    # Flatten so CE sees each dx/dy as separate
    logits_flat = logits.reshape(B * 8, num_classes)  # (B*8,21)
    targets_flat = target.reshape(B * 8)  # (B*8)

    loss = ce(logits_flat, targets_flat)
    return loss


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_offset_sign(image_dir, window_size=64, margin=16, disp_range=(-16, 16)):
    # Pick random image
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_paths:
        print("❌ No images found in", image_dir)
        return

    img_path = random.choice(image_paths)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image:", img_path)
        return

    # Generate pair using your pipeline
    pair, offsets, src_corners, warped_true = generate_pair(
        img, window_size=window_size, margin=margin, disp_range=disp_range
    )
    dst_corners = src_corners + offsets

    # Reconstruct warps using both offset signs
    H_plus = cv2.getPerspectiveTransform(src_corners, dst_corners)
    H_minus = cv2.getPerspectiveTransform(src_corners, src_corners - offsets)

    warped_plus = cv2.warpPerspective(img, H_plus, (img.shape[1], img.shape[0]))
    warped_minus = cv2.warpPerspective(img, H_minus, (img.shape[1], img.shape[0]))

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # 1️⃣ True warped image
    axes[0].imshow(warped_true, cmap='gray')
    axes[0].add_patch(plt.Polygon(src_corners, fill=False, edgecolor='green', lw=2, label='src'))
    axes[0].add_patch(plt.Polygon(dst_corners, fill=False, edgecolor='blue', lw=2, label='dst (GT)'))
    axes[0].set_title("True warped image (from generate_pair)")
    axes[0].legend()
    axes[0].axis('off')

    # 2️⃣ Reconstructed with +offsets
    axes[1].imshow(warped_plus, cmap='gray')
    axes[1].add_patch(plt.Polygon(src_corners, fill=False, edgecolor='green', lw=2))
    axes[1].add_patch(plt.Polygon(dst_corners, fill=False, edgecolor='red', lw=2))
    axes[1].set_title("Reconstructed warp (+offsets)")
    axes[1].axis('off')

    # 3️⃣ Reconstructed with -offsets
    axes[2].imshow(warped_minus, cmap='gray')
    axes[2].add_patch(plt.Polygon(src_corners, fill=False, edgecolor='green', lw=2))
    axes[2].add_patch(plt.Polygon(dst_corners, fill=False, edgecolor='red', lw=2))
    axes[2].set_title("Reconstructed warp (-offsets)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Numeric comparison for confirmation
    diff_plus = np.mean(np.abs(warped_true.astype(np.float32) - warped_plus.astype(np.float32)))
    diff_minus = np.mean(np.abs(warped_true.astype(np.float32) - warped_minus.astype(np.float32)))

    print(f"Mean pixel difference (true vs +offsets): {diff_plus:.2f}")
    print(f"Mean pixel difference (true vs -offsets): {diff_minus:.2f}")

    if diff_minus < diff_plus:
        print("⚠️ Offsets likely need to be NEGATED during training.")
    else:
        print("✅ Offsets appear to have the correct sign.")


def visualize_regression_result(model, image, window_size=64, margin=16, disp_range=(-16, 16),
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()

    # Generate pair from your existing pipeline
    pair, offsets_gt, corners_pair, image_pair = generate_pair(
        image, window_size=window_size, margin=margin, disp_range=disp_range
    )

    # Normalize / minimal checks
    pair = np.asarray(pair, dtype=np.float32)  # H x W x 2
    offsets_gt = np.asarray(offsets_gt, dtype=np.float32).squeeze()
    corners_pair = np.asarray(corners_pair, dtype=np.float32)

    # Extract src_corners and dst_corners from corners_pair (common layouts)
    if corners_pair.ndim == 3 and corners_pair.shape[2] == 2:
        src_corners = corners_pair[:, :, 0]
        dst_corners_gt = corners_pair[:, :, 1]
    elif corners_pair.ndim == 3 and corners_pair.shape[0] == 2 and corners_pair.shape[1] == 4:
        src_corners = corners_pair[0]
        dst_corners_gt = corners_pair[1]
    else:
        raise ValueError(f"Unexpected corners_pair shape: {corners_pair.shape}")

    dst_corners_gt = src_corners + offsets_gt  # ground-truth displaced corners

    # Prepare tensor
    pair_tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Predict offsets
    with torch.no_grad():
        pred_offsets = model(pair_tensor).cpu().numpy().reshape(4, 2)

    dst_corners_pred = src_corners + pred_offsets  # predicted displaced corners

    # Extract patches
    patch_original = pair[:, :, 0]  # Original patch
    patch_warped = pair[:, :, 1]    # Warped patch (GT transformation)

    # Create predicted warped patch (same process as GT, but with predicted offsets)
    h, w = patch_original.shape

    # Original patch corner positions (in patch coordinate space)
    patch_corners_orig = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Displaced corners using predicted offsets
    patch_corners_pred = patch_corners_orig + pred_offsets

    # Compute homography from original corners to predicted displaced corners
    H_pred, _ = cv2.findHomography(patch_corners_orig, patch_corners_pred, method=0)
    H_pred_inv = np.linalg.inv(H_pred)

    # Warp the original patch with the inverse (same as how GT warped patch is created)
    patch_warped_pred = cv2.warpPerspective(
        (patch_original * 255).astype(np.uint8),
        H_pred_inv,
        (w, h)
    ) / 255.0

    # Compute errors
    error = pred_offsets - offsets_gt
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    # --- Visualization ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Row 1: Full image with wireframes
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(image, cmap='gray')

    # Plot polygons: source, GT, predicted
    ax_full.add_patch(plt.Polygon(src_corners, fill=False, edgecolor='lime', lw=2, label='Source'))
    ax_full.add_patch(plt.Polygon(dst_corners_gt, fill=False, edgecolor='cyan', lw=2, label='Warped (GT)'))
    ax_full.add_patch(plt.Polygon(dst_corners_pred, fill=False, edgecolor='red', lw=2, label='Predicted'))

    # Draw corner points and labels
    for i, (x, y) in enumerate(src_corners):
        ax_full.plot(x, y, 'go', markersize=6)
        ax_full.text(x + 3, y - 3, str(i), color='lime', fontsize=10, weight='bold')

    for (x, y) in dst_corners_pred:
        ax_full.plot(x, y, 'ro', markersize=6)

    for (x, y) in dst_corners_gt:
        ax_full.plot(x, y, 'co', markersize=6)

    ax_full.legend(loc='upper right', fontsize=10)
    ax_full.set_title(f"Homography Regression Visualization\nGreen=Source, Cyan=GT Warped, Red=Predicted\nRMSE: {rmse:.2f}px, MAE: {mae:.2f}px",
                     fontsize=12, fontweight='bold')
    ax_full.axis("off")

    # Row 2: Three patches
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(patch_original, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Original Patch', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(patch_warped, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Warped Patch (GT)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(patch_warped_pred, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Warped Patch (Predicted)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Optional: print numeric details
    print("Ground Truth Offsets:\n", np.round(offsets_gt, 2))
    print("Predicted Offsets:\n", np.round(pred_offsets, 2))
    print("Mean abs error per corner:", np.mean(np.abs(pred_offsets - offsets_gt), axis=0).round(3))


def visualize_classification_result(
        model,
        image,
        window_size=64,
        margin=16,
        disp_range=(-16, 16),
        soft_decode=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.eval()

    pair, offsets_gt, corners_pair, image_pair = generate_pair(
        image, window_size=window_size, margin=margin, disp_range=disp_range
    )

    # Normalize / minimal checks
    pair = np.asarray(pair, dtype=np.float32)  # H x W x 2
    offsets_gt = np.asarray(offsets_gt, dtype=np.float32).squeeze()
    corners_pair = np.asarray(corners_pair, dtype=np.float32)

    # Extract src_corners and dst_corners from corners_pair (common layouts)
    if corners_pair.ndim == 3 and corners_pair.shape[2] == 2:
        src_corners = corners_pair[:, :, 0]
        dst_corners_gt = corners_pair[:, :, 1]
    elif corners_pair.ndim == 3 and corners_pair.shape[0] == 2 and corners_pair.shape[1] == 4:
        src_corners = corners_pair[0]
        dst_corners_gt = corners_pair[1]
    else:
        raise ValueError(f"Unexpected corners_pair shape: {corners_pair.shape}")

    dst_corners_gt = src_corners + offsets_gt  # true warped corners

    # === Prepare tensor ===
    pair_tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # === Forward pass ===
    with torch.no_grad():
        preds = model(pair_tensor)  # (1, 21, 8) logits
        pred_offsets = classes_to_offsets(preds, disp_range, soft=soft_decode).cpu().numpy().reshape(4, 2)

    dst_corners_pred = src_corners + pred_offsets  # predicted warped corners

    # Extract patches
    patch_original = pair[:, :, 0]  # Original patch
    patch_warped = pair[:, :, 1]    # Warped patch (GT transformation)

    # Create predicted warped patch (same process as GT, but with predicted offsets)
    h, w = patch_original.shape

    # Original patch corner positions (in patch coordinate space)
    patch_corners_orig = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Displaced corners using predicted offsets
    patch_corners_pred = patch_corners_orig + pred_offsets

    # Compute homography from original corners to predicted displaced corners
    H_pred, _ = cv2.findHomography(patch_corners_orig, patch_corners_pred, method=0)
    H_pred_inv = np.linalg.inv(H_pred)

    # Warp the original patch with the inverse (same as how GT warped patch is created)
    patch_warped_pred = cv2.warpPerspective(
        (patch_original * 255).astype(np.uint8),
        H_pred_inv,
        (w, h)
    ) / 255.0

    # Compute errors
    error = pred_offsets - offsets_gt
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    # === Visualization ===
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Row 1: Full image with wireframes
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(image, cmap='gray')

    # Draw polygons
    ax_full.add_patch(plt.Polygon(src_corners, fill=False, edgecolor='lime', lw=2, label='Source'))
    ax_full.add_patch(plt.Polygon(dst_corners_gt, fill=False, edgecolor='cyan', lw=2, label='GT Warped'))
    ax_full.add_patch(plt.Polygon(dst_corners_pred, fill=False, edgecolor='red', lw=2, label='Predicted'))

    # Draw corner markers
    for i, (x, y) in enumerate(src_corners):
        ax_full.plot(x, y, 'go', markersize=6)
        ax_full.text(x + 3, y - 3, str(i), color='lime', fontsize=10, weight='bold')

    for (x, y) in dst_corners_gt:
        ax_full.plot(x, y, 'co', markersize=5)

    for (x, y) in dst_corners_pred:
        ax_full.plot(x, y, 'ro', markersize=5)

    ax_full.legend(loc='upper right', fontsize=10)
    decode_mode = "Soft" if soft_decode else "Hard"
    ax_full.set_title(f"Homography Classification Visualization ({decode_mode})\nGreen=Src, Cyan=GT, Red=Predicted\nRMSE: {rmse:.2f}px, MAE: {mae:.2f}px",
                     fontsize=12, fontweight='bold')
    ax_full.axis("off")

    # Row 2: Three patches
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(patch_original, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Original Patch', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(patch_warped, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Warped Patch (GT)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(patch_warped_pred, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Warped Patch (Predicted)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # === Prediction Maps Visualization ===
    # Compute probability maps for each corner showing confidence distribution
    with torch.no_grad():
        # Model outputs (1, 8, 21), transpose to (1, 21, 8) to match reference code format
        preds_transposed = preds.transpose(1, 2)  # (1, 21, 8)
        preds_softmax = torch.nn.Softmax(dim=1)(preds_transposed)  # Apply softmax over classes (dim=1)

        # Combine x and y coordinate probabilities to create 2D probability distributions
        # Even indices (0, 2, 4, 6) are x coordinates, odd indices (1, 3, 5, 7) are y coordinates
        # pred_maps[i, j, k] = P(x_offset=i) * P(y_offset=j) for corner k
        pred_maps = (preds_softmax[:, :, ::2].reshape(-1, 21, 1, 4) *
                     preds_softmax[:, :, 1::2].reshape(-1, 1, 21, 4)).cpu().numpy()

    # Create prediction maps figure
    fig_maps, axes_maps = plt.subplots(1, 4, figsize=(16, 4))

    min_disp, max_disp = disp_range
    extent = [min_disp, max_disp, max_disp, min_disp]  # [left, right, bottom, top]

    for corner_idx in range(4):
        ax = axes_maps[corner_idx]

        # Display the probability heatmap showing confidence distribution
        im = ax.imshow(pred_maps[0, :, :, corner_idx], extent=extent, cmap='viridis', aspect='auto', interpolation='bilinear')

        # Mark the ground truth offset (from generate_pair) with green circle
        # Note: reference code plots as (y, x) so we swap the indices
        gt_x = offsets_gt[corner_idx, 1]  # y-coordinate
        gt_y = offsets_gt[corner_idx, 0]  # x-coordinate
        ax.plot(gt_x, gt_y, 'go', markersize=10, markerfacecolor='none', markeredgewidth=2.5, label='GT')

        # Mark the predicted offset (from model) with red plus
        # Note: reference code plots as (y, x) so we swap the indices
        pred_x = pred_offsets[corner_idx, 1]  # y-coordinate
        pred_y = pred_offsets[corner_idx, 0]  # x-coordinate
        ax.plot(pred_x, pred_y, 'r+', markersize=12, markeredgewidth=2.5, label='Predicted')

        ax.set_xlabel('Y Offset (px)', fontsize=10)
        ax.set_ylabel('X Offset (px)', fontsize=10)
        ax.set_title(f'Corner {corner_idx}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8)

        # Add colorbar showing probability/confidence
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Confidence', fontsize=9)

    plt.suptitle(f'Prediction Confidence Maps ({decode_mode} Decode)\nGreen Circle = Ground Truth, Red Plus = Predicted',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # === Numeric comparison ===
    print("Ground Truth Offsets:\n", np.round(offsets_gt, 2))
    print("Predicted Offsets:\n", np.round(pred_offsets, 2))
    print("Mean Abs Error per corner [px]:", np.mean(np.abs(pred_offsets - offsets_gt), axis=0).round(3))

    mean_abs_error = np.mean(np.abs(pred_offsets - offsets_gt))
    print(f"Overall Mean Abs Error: {mean_abs_error:.3f} px")

    return mean_abs_error


def visualize_classification_result_dataloader(
        model,
        dataloader,
        num_samples=3,
        disp_range=(-16, 16),
        soft_decode=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.eval()

    # Get one batch
    pairs, offsets_gt = next(iter(dataloader))
    batch_size = min(num_samples, pairs.shape[0])

    # Move to device and predict
    pairs = pairs.to(device)
    with torch.no_grad():
        preds = model(pairs)  # (B, 8, 21)
        pred_offsets = classes_to_offsets(preds, disp_range, soft=soft_decode)

    # Move back to CPU
    pairs = pairs.cpu().numpy()
    offsets_gt = offsets_gt.cpu().numpy()
    pred_offsets = pred_offsets.cpu().numpy()

    # Create subplots
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # Extract patches
        orig_patch = pairs[i, 0]  # (64, 64)
        warped_patch = pairs[i, 1]  # (64, 64)

        # Reshape offsets
        offsets_gt_i = offsets_gt[i].reshape(4, 2)
        offsets_pred_i = pred_offsets[i].reshape(4, 2)

        # Define patch corners (local coordinates)
        patch_corners = np.array([
            [0, 0],
            [63, 0],
            [63, 63],
            [0, 63]
        ], dtype=np.float32)

        # Reconstruct destination corners
        dst_corners_gt = patch_corners + offsets_gt_i
        dst_corners_pred = patch_corners + offsets_pred_i

        # --- Plot 1: Original + GT ---
        ax = axes[i, 0]
        ax.imshow(orig_patch, cmap='gray')
        ax.add_patch(plt.Polygon(patch_corners, fill=False, edgecolor='lime', lw=2, label='Source'))
        ax.add_patch(plt.Polygon(dst_corners_gt, fill=False, edgecolor='cyan', lw=2, label='GT Warp'))

        for j in range(4):
            ax.arrow(patch_corners[j, 0], patch_corners[j, 1],
                     offsets_gt_i[j, 0], offsets_gt_i[j, 1],
                     head_width=2, head_length=2, fc='cyan', ec='cyan', alpha=0.7)
            ax.plot(patch_corners[j, 0], patch_corners[j, 1], 'go', markersize=8)
            ax.text(patch_corners[j, 0] + 2, patch_corners[j, 1] - 2,
                    str(j), color='lime', fontsize=10, weight='bold')

        ax.set_title(f'Sample {i + 1}: Original + GT\nAvg offset: {np.abs(offsets_gt_i).mean():.2f}px')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')

        # --- Plot 2: Original + Predicted ---
        ax = axes[i, 1]
        ax.imshow(orig_patch, cmap='gray')
        ax.add_patch(plt.Polygon(patch_corners, fill=False, edgecolor='lime', lw=2, label='Source'))
        ax.add_patch(plt.Polygon(dst_corners_pred, fill=False, edgecolor='red', lw=2, label='Pred Warp'))

        for j in range(4):
            ax.arrow(patch_corners[j, 0], patch_corners[j, 1],
                     offsets_pred_i[j, 0], offsets_pred_i[j, 1],
                     head_width=2, head_length=2, fc='red', ec='red', alpha=0.7)
            ax.plot(patch_corners[j, 0], patch_corners[j, 1], 'go', markersize=8)

        decode_type = "Soft" if soft_decode else "Hard"
        ax.set_title(f'Predicted ({decode_type})\nAvg offset: {np.abs(offsets_pred_i).mean():.2f}px')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')

        # --- Plot 3: Warped + Errors ---
        ax = axes[i, 2]
        ax.imshow(warped_patch, cmap='gray')

        # Calculate errors
        corner_errors = np.linalg.norm(offsets_pred_i - offsets_gt_i, axis=1)

        for j in range(4):
            # GT corners
            ax.plot(dst_corners_gt[j, 0], dst_corners_gt[j, 1], 'co', markersize=8, label='GT' if j == 0 else '')
            # Predicted corners
            ax.plot(dst_corners_pred[j, 0], dst_corners_pred[j, 1], 'rx', markersize=10, mew=2,
                    label='Pred' if j == 0 else '')
            # Error line
            ax.plot([dst_corners_gt[j, 0], dst_corners_pred[j, 0]],
                    [dst_corners_gt[j, 1], dst_corners_pred[j, 1]],
                    'y--', linewidth=1, alpha=0.7)
            # Error text
            ax.text(dst_corners_pred[j, 0] + 3, dst_corners_pred[j, 1] - 3,
                    f'{corner_errors[j]:.1f}', color='yellow', fontsize=9, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

        mae = np.mean(np.abs(offsets_pred_i - offsets_gt_i))
        rmse = np.sqrt(np.mean((offsets_pred_i - offsets_gt_i) ** 2))

        ax.set_title(f'Warped + Errors\nMAE: {mae:.3f}px | RMSE: {rmse:.3f}px')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Print metrics
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION RESULTS ({decode_type} decoding)")
    print("=" * 60)
    for i in range(batch_size):
        offsets_gt_i = offsets_gt[i].reshape(4, 2)
        offsets_pred_i = pred_offsets[i].reshape(4, 2)

        print(f"\nSample {i + 1}:")
        print("  Corner | GT Offset (Δx, Δy) | Pred Offset (Δx, Δy) | Error (px)")
        print("  " + "-" * 70)
        for j in range(4):
            error = np.linalg.norm(offsets_pred_i[j] - offsets_gt_i[j])
            print(f"    {j}    | ({offsets_gt_i[j, 0]:+6.2f}, {offsets_gt_i[j, 1]:+6.2f}) | "
                  f"({offsets_pred_i[j, 0]:+6.2f}, {offsets_pred_i[j, 1]:+6.2f}) | {error:6.3f}")

        mae = np.mean(np.abs(offsets_pred_i - offsets_gt_i))
        rmse = np.sqrt(np.mean((offsets_pred_i - offsets_gt_i) ** 2))
        print(f"  Overall: MAE={mae:.3f}px, RMSE={rmse:.3f}px")


def test_offset_class_conversion():
    num_classes = 21
    disp_range = (-16, 16)

    # Test with known values and random values
    test_cases = [
        torch.tensor([[-16., -8., 0., 8., 16., 1.2, -3.7, 15.9]]),  # Known
        torch.rand(1, 8) * 32 - 16  # Random
    ]

    for i, offsets_orig in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}\nTest case {i}")
        print(f"Original offsets: {offsets_orig}")

        # Forward: offsets → classes
        class_idx = offsets_to_class_indices(offsets_orig, num_classes, disp_range)
        print(f"Class indices: {class_idx}")

        # Create one-hot logits
        logits = torch.zeros(1, 8, num_classes)
        logits[0, torch.arange(8), class_idx[0].long()] = 100.0

        # Backward: classes → offsets (hard)
        recovered_hard = classes_to_offsets(logits, disp_range, soft=False)
        hard_error = torch.abs(recovered_hard - offsets_orig).max().item()
        print(f"Recovered (hard): {recovered_hard}")
        print(f"Hard max error: {hard_error:.6f} px")

        # Backward: classes → offsets (soft)
        logits_noisy = logits + torch.randn_like(logits) * 0.1
        recovered_soft = classes_to_offsets(logits_noisy, disp_range, soft=True)
        soft_error = torch.abs(recovered_soft - offsets_orig).max().item()
        print(f"Recovered (soft): {recovered_soft}")
        print(f"Soft max error: {soft_error:.6f} px")

    print("\n" + "=" * 60)
