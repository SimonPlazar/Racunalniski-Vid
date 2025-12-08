import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.ndimage import maximum_filter
from Generator import generate_synthetic_image, generate_random_homography


# ============================================================
# RESNET
# ============================================================
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut, če se spremeni dimenzija
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


# Encoder

class KeypointEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            ResNetBlock(1, 64),
            ResNetBlock(64, 64),
            nn.MaxPool2d(2)  # ↓ 1/2
        )

        self.layer2 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
            nn.MaxPool2d(2)  # ↓ 1/4
        )

        self.layer3 = nn.Sequential(
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            nn.MaxPool2d(2)  # ↓ 1/8
        )

        self.layer4 = nn.Sequential(
            ResNetBlock(128, 128),
            ResNetBlock(128, 128),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # velikost: (B,128,H/8,W/8)


# Decoder

class KeypointDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(256, 65, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return x  # (B,65,H/8,W/8)


# Full Model

class KeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = KeypointEncoder()
        self.decoder = KeypointDecoder()

    def forward(self, x, return_logits=False):
        """
        Forward pass of the keypoint detection network.

        Args:
            x: Input image (B, 1, H, W)
            return_logits: If True, return raw logits for training.
                          If False, return probability heatmap for inference.

        Returns:
            If return_logits=True: (B, 65, H/8, W/8) - raw logits for each position
            If return_logits=False: (B, 1, H, W) - probability heatmap
        """
        x = self.encoder(x)  # (B,128,H/8,W/8)
        logits = self.decoder(x)  # (B,65,H/8,W/8)

        if return_logits:
            return logits

        # For inference: convert to probability heatmap
        # softmax po kanalih
        heat = torch.softmax(logits, dim=1)  # (B,65,H/8,W/8)

        # odstranimo zadnji kanal = razred "ni točke"
        heat = heat[:, :64, :, :]  # (B,64,H/8,W/8)

        # preoblikovanje v H×W
        # depth_to_space 8×8
        heat = torch.nn.functional.pixel_shuffle(heat, upscale_factor=8)
        # rezultat: (B,1,H,W)

        return heat


# ============================================================
# KEYPOINT DETECTION UTILITIES
# ============================================================

def process_output_torch(logits, threshold=0.015):
    """
    Verzija z uporabo PyTorch PixelShuffle (ekvivalentno TensorFlow depth_to_space).

    Args:
        logits: (B, 65, H/8, W/8) izhod mreže
        threshold: prag za detekcijo točk

    Returns:
        heat_map: (B, 1, H, W) verjetnostna mapa
        keypoints: list of (N, 2) numpy arrays z (x, y) koordinatami
    """
    B, C, Hc, Wc = logits.shape
    assert C == 65, f"Expected 65 channels, got {C}"

    # 1. Softmax po kanalih
    heat = F.softmax(logits, dim=1)  # (B, 65, H/8, W/8)

    # 2. Odstranimo zadnji kanal
    heat = heat[:, :64, :, :]  # (B, 64, H/8, W/8)

    # 3. PixelShuffle (depth_to_space)
    heat = F.pixel_shuffle(heat, upscale_factor=8)  # (B, 1, H, W)

    # 4. Squeeze če želimo (B, H, W)
    heat_squeezed = heat.squeeze(1)  # (B, H, W)

    # 5. Iskanje lokalnih maksimumov
    keypoints = []
    heat_np = heat_squeezed.detach().cpu().numpy()

    for i in range(B):
        heat_i = heat_np[i]

        # Lokalni maksimumi
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(heat_i, size=3)
        detected = (heat_i == local_max) & (heat_i > threshold)

        # Koordinate
        ys, xs = np.where(detected)
        kpts = np.stack([xs, ys], axis=1) if len(xs) > 0 else np.empty((0, 2))
        keypoints.append(kpts)

    return heat, keypoints


def prepare_training_target_basic(keypoints_list, image_shape, cell_size=8, sigma=1.0):
    """
    Correct SuperPoint-style target generation.

    Args:
        keypoints_list : list of (N, 2) arrays of keypoints (x, y)
        image_shape    : (H, W)
        cell_size      : normally 8 for SuperPoint
        sigma          : stddev for soft Gaussian inside each cell (1.0 recommended)

    Returns:
        target tensor (B, 65, H/8, W/8)
    """

    H, W = image_shape
    Hc, Wc = H // cell_size, W // cell_size

    B = len(keypoints_list)
    target = np.zeros((B, 65, Hc, Wc), dtype=np.float32)

    # Precompute gaussian "pulse" inside an 8×8 cell
    # centered at (sub_y, sub_x)
    yy, xx = np.meshgrid(np.arange(cell_size), np.arange(cell_size), indexing='ij')

    for b, keypoints in enumerate(keypoints_list):

        if len(keypoints) == 0:
            # no keypoints → whole map stays "no point"
            continue

        for (x_f, y_f) in keypoints:
            # ensure within bounds
            if x_f < 0 or x_f >= W or y_f < 0 or y_f >= H:
                continue

            # DO NOT ROUND — SuperPoint uses FLOOR (integer pixel)
            x = int(np.floor(x_f))
            y = int(np.floor(y_f))

            # Which cell?
            cell_x = x // cell_size
            cell_y = y // cell_size

            # Offset inside cell
            sub_x = x % cell_size
            sub_y = y % cell_size

            # Compute 8×8 Gaussian centered at (sub_y, sub_x)
            dist2 = (yy - sub_y) ** 2 + (xx - sub_x) ** 2
            gaussian = np.exp(-0.5 * dist2 / (sigma * sigma))

            # Convert the 8×8 cell into a 64-D vector (flatten in row-major order)
            gaussian_flat = gaussian.reshape(-1)  # (64,)

            # **Write into (64, Hc, Wc) target**
            # If multiple points fall in same cell → keep the one with higher peak
            existing = target[b, :64, cell_y, cell_x]
            if existing.sum() > 0:
                # Keep the one whose center is closest to the cell center
                # i.e. larger gaussian peak
                if gaussian_flat.max() > existing.max():
                    target[b, :64, cell_y, cell_x] = gaussian_flat
            else:
                target[b, :64, cell_y, cell_x] = gaussian_flat

        # Now fill channel 64 = "no point" class
        occupied = target[b, :64, :, :].sum(axis=0) > 0
        target[b, 64, :, :] = (~occupied).astype(np.float32)

    return torch.from_numpy(target)


# Helper function to extract keypoint positions from target tensor
def extract_keypoints_from_target(target_tensor):
    """
    Extract keypoint positions from target tensor.

    Args:
        target_tensor: (65, H/8, W/8) target tensor

    Returns:
        keypoint_positions: List of [x, y] positions in image coordinates
    """
    target_np = target_tensor.numpy() if hasattr(target_tensor, 'numpy') else target_tensor

    Hc, Wc = target_np.shape[1], target_np.shape[2]  # Downsampled dimensions (32, 32)
    keypoint_positions = []

    for h in range(Hc):
        for w in range(Wc):
            # Check if any of the 64 spatial channels are active (not the "no point" channel)
            cell_data = target_np[:64, h, w]
            if cell_data.sum() > 0:
                # Find which position in the 8x8 cell is active
                active_idx = np.argmax(cell_data)
                # Convert to image coordinates
                sub_y = active_idx // 8
                sub_x = active_idx % 8
                y = h * 8 + sub_y
                x = w * 8 + sub_x
                keypoint_positions.append([x, y])

    return np.array(keypoint_positions) if len(keypoint_positions) > 0 else np.array([]).reshape(0, 2)


# def detect_local_maxima(heatmap, threshold=0.015, neighborhood_size=9):
#     """
#     Detect local maxima in heatmap using maximum filter.
#
#     Args:
#         heatmap: (H, W) numpy array
#         threshold: Detection threshold
#
#     Returns:
#         peaks: (N, 2) array of [x, y] coordinates
#     """
#     local_max = maximum_filter(heatmap, size=neighborhood_size)
#     detected = (heatmap == local_max) & (heatmap > threshold)
#
#     ys, xs = np.where(detected)
#     peaks = np.stack([xs, ys], axis=1) if len(xs) > 0 else np.empty((0, 2))
#
#     return peaks

def detect_local_maxima(heatmap, threshold=0.015, nms_size=9, refine_radius=7):
    """
    Two-stage: Peak detection → weighted refinement.

    Args:
        heatmap: (H, W) numpy array
        threshold: Detection threshold
        nms_size: Neighborhood for initial peak detection (larger for blurred heatmaps)
        refine_radius: Radius for weighted average refinement

    Returns:
        peaks: (N, 2) array of [x, y] coordinates
    """
    # Stage 1: Find peaks using maximum filter (robust to blur)
    local_max = maximum_filter(heatmap, size=nms_size)
    detected = (heatmap == local_max) & (heatmap > threshold)
    ys, xs = np.where(detected)

    if len(xs) == 0:
        return np.empty((0, 2))

    # Stage 2: Refine each peak using weighted average
    H, W = heatmap.shape
    peaks = []

    for x, y in zip(xs, ys):
        # Define local window
        y_min = max(0, y - refine_radius)
        y_max = min(H, y + refine_radius + 1)
        x_min = max(0, x - refine_radius)
        x_max = min(W, x + refine_radius + 1)

        # Extract local region
        local_heatmap = heatmap[y_min:y_max, x_min:x_max]
        local_mask = local_heatmap > threshold

        if local_mask.sum() == 0:
            peaks.append([x, y])
            continue

        # Weighted average in local region
        yy, xx = np.meshgrid(np.arange(y_min, y_max),
                             np.arange(x_min, x_max), indexing='ij')
        weights = local_heatmap[local_mask]
        x_refined = np.average(xx[local_mask], weights=weights)
        y_refined = np.average(yy[local_mask], weights=weights)

        peaks.append([x_refined, y_refined])

    return np.array(peaks)


def homography_adaptation(model, image, num_iter=99, threshold=0.015, padding_ratio=0.3):
    """
    Homography adaptation WITHOUT border artifacts using padding strategy.

    Strategy: Pad the input image, apply homographies to padded version,
    then crop back to original size. This ensures the original region
    always has valid pixels (no black borders).

    Args:
        model: KeypointNet model
        image: (1, 1, H, W) tensor (grayscale, normalized [0,1])
        num_iter: Number of random homographies (default 99)
        threshold: Detection threshold for local maxima
        padding_ratio: Ratio of padding to add (0.3 = 30% padding on each side)

    Returns:
        averaged: (1, 1, H, W) averaged heatmap
        keypoints: (N, 2) array of detected keypoints [x, y]
    """
    device = image.device
    B, C, H, W = image.shape

    assert B == 1 and C == 1, "Expected single grayscale image (1, 1, H, W)"

    # Calculate padding size
    pad_h = int(H * padding_ratio)
    pad_w = int(W * padding_ratio)

    # Pad image (replicate border pixels to avoid black edges)
    image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    _, _, H_pad, W_pad = image_padded.shape

    print(f"Original size: {H}x{W}, Padded size: {H_pad}x{W_pad}")

    # Accumulator for heatmaps (original size)
    accumulated_heatmap = torch.zeros((1, 1, H, W), device=device)

    model.eval()
    with torch.no_grad():
        # Iteration 0: Original image (identity homography)
        print("Processing original image...")
        heatmap_orig = model(image, return_logits=False)  # (1, 1, H, W)
        accumulated_heatmap += heatmap_orig

        # Iterations 1 to num_iter: Random homographies
        for i in range(num_iter):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_iter} homographies")

            # Generate random homography for PADDED image
            H_forward = generate_random_homography(W_pad, H_pad)
            H_inverse = np.linalg.inv(H_forward)

            # Convert to tensor
            H_inv_tensor = torch.from_numpy(H_inverse).float().to(device)

            # **STEP 1**: Warp PADDED image
            warped_padded = warp_tensor_with_homography_simple(
                image_padded, H_forward, (H_pad, W_pad)
            )

            # **STEP 2**: Crop warped image to center (original size)
            warped_center = warped_padded[:, :, pad_h:pad_h + H, pad_w:pad_w + W]

            # **STEP 3**: Get heatmap from center crop
            heatmap_warped = model(warped_center, return_logits=False)  # (1, 1, H, W)

            # **STEP 4**: Pad heatmap back to full size for inverse warping
            heatmap_padded = F.pad(heatmap_warped, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

            # **STEP 5**: Warp heatmap BACK using inverse homography
            heatmap_unwarped_padded = warp_tensor_with_homography_simple(
                heatmap_padded, H_inverse, (H_pad, W_pad)
            )

            # **STEP 6**: Crop unwarped heatmap to center
            heatmap_unwarped = heatmap_unwarped_padded[:, :, pad_h:pad_h + H, pad_w:pad_w + W]

            # **STEP 7**: Accumulate
            accumulated_heatmap += heatmap_unwarped

    # Average over all iterations
    averaged = accumulated_heatmap / (num_iter + 1)

    # Extract keypoints from averaged heatmap
    heatmap_np = averaged.squeeze().cpu().numpy()
    keypoints = detect_local_maxima(heatmap_np, threshold)

    print(f"✓ Detected {len(keypoints)} keypoints after homography adaptation")

    return averaged, keypoints


def warp_tensor_with_homography_simple(tensor, H, output_size):
    """
    Simplified homography warping without valid mask tracking.

    Args:
        tensor: (B, C, H, W) tensor to warp
        H: (3, 3) homography matrix (NumPy or PyTorch)
        output_size: (H_out, W_out) output dimensions

    Returns:
        warped: (B, C, H_out, W_out) warped tensor
    """
    device = tensor.device
    B, C, H_in, W_in = tensor.shape
    H_out, W_out = output_size

    # Convert H to tensor if needed
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float().to(device)

    # Create normalized meshgrid for output image
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, H_out - 1, H_out, device=device),
        torch.linspace(0, W_out - 1, W_out, device=device),
        indexing='ij'
    )

    # Stack into homogeneous coordinates (H_out, W_out, 3)
    ones = torch.ones_like(x_grid)
    grid_homogeneous = torch.stack([x_grid, y_grid, ones], dim=-1)

    # Apply homography: p_src = H @ p_dst
    grid_warped = torch.matmul(grid_homogeneous, H.T)  # (H_out, W_out, 3)

    # Convert from homogeneous to Cartesian
    grid_warped_xy = grid_warped[..., :2] / (grid_warped[..., 2:3] + 1e-8)

    # Normalize to [-1, 1] for grid_sample
    grid_normalized = torch.zeros_like(grid_warped_xy)
    grid_normalized[..., 0] = 2.0 * grid_warped_xy[..., 0] / (W_in - 1) - 1.0
    grid_normalized[..., 1] = 2.0 * grid_warped_xy[..., 1] / (H_in - 1) - 1.0

    # Add batch dimension
    grid_normalized = grid_normalized.unsqueeze(0).expand(B, -1, -1, -1)

    # Apply warping with REPLICATE padding (no black borders!)
    warped = F.grid_sample(
        tensor,
        grid_normalized,
        mode='bilinear',
        padding_mode='border',  # Key change: replicate border instead of zeros
        align_corners=True
    )

    return warped


# ============================================================
# DATASET AUGMENTATION FUNCTIONS
# ============================================================

from torch.utils.data import Dataset
import cv2


def apply_homography_augmentation(img, keypoints, image_shape, max_retries=10):
    """
    Apply random homography transformation to image and keypoints.

    Args:
        img: Grayscale image (H, W)
        keypoints: (N, 2) array of keypoint coordinates
        image_shape: (H, W) tuple
        max_retries: Maximum number of retry attempts

    Returns:
        Augmented image and keypoints (or original if all attempts fail)
    """
    from Generator import generate_random_homography, apply_homography

    H, W = image_shape
    min_keypoints = min(3, len(keypoints)) if len(keypoints) > 0 else 0

    for attempt in range(max_retries):
        # Generate random homography
        homography_matrix = generate_random_homography(W, H)

        # Apply to image and keypoints
        img_warped, keypoints_warped = apply_homography(
            img, keypoints, homography_matrix, W, H
        )

        # Accept if we retained enough keypoints
        if len(keypoints_warped) >= min_keypoints:
            return img_warped, keypoints_warped

    # If all retries failed, return original
    return img, keypoints


def apply_photometric_augmentation(img):
    """
    Apply photometric augmentations: brightness and contrast adjustments.
        img: Grayscale image (H, W) as uint8

    Returns:
        Augmented image
    """
    # Random brightness adjustment
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return img


def apply_geometric_augmentation(img, keypoints, image_shape):
    """
    Apply geometric augmentations: horizontal and vertical flips.

    Args:
        img: Grayscale image (H, W)
        keypoints: (N, 2) array of keypoint coordinates
        image_shape: (H, W) tuple

    Returns:
        Augmented image and keypoints
    """
    H, W = image_shape

    # Random horizontal flip (50% chance)
    if np.random.rand() < 0.5:
        # Flip the image horizontally (left-right)
        img = np.fliplr(img)
        # Flip keypoint x-coordinates
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, 0] = (W - 1) - keypoints[:, 0]

    # Random vertical flip (50% chance)
    if np.random.rand() < 0.5:
        # Flip the image vertically (up-down)
        img = np.flipud(img)
        # Flip keypoint y-coordinates
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, 1] = (H - 1) - keypoints[:, 1]

    return img, keypoints


# ============================================================
# KEYPOINT DATASET
# ============================================================


class KeypointDataset(Dataset):
    """
    Dataset for keypoint detection training with iteration-based approach.

    Supports infinite cycling through samples for iteration-based training.

    Modes:
    - pregenerate=True: Pre-generate base images, apply augmentation on-the-fly (RECOMMENDED)
    - pregenerate=False: Generate completely new images each iteration (slower, max variety)
    - load_from_file: Load pre-generated samples from file (path to .npz file)
    """

    def __init__(
            self,
            num_samples,
            image_shape,
            generate_fn=None,
            generate_kwargs=None,
            use_homography_augment=True,
            use_photometric_augment=True,
            use_geometric_augment=True,
            pregenerate=True,
            load_from_file=None
    ):
        """
        Args:
            num_samples: Number of base samples (if pregenerate=True)
            image_shape: (H, W) - must be divisible by 8
            generate_fn: Function to generate synthetic images
            generate_kwargs: Kwargs for generate_fn
            use_homography_augment: Apply random homography
            use_photometric_augment: Apply brightness/contrast
            use_geometric_augment: Apply flips
            pregenerate: Pre-generate base samples
            load_from_file: Path to .npz file with pre-generated data
        """
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.use_homography_augment = use_homography_augment
        self.use_photometric_augment = use_photometric_augment
        self.use_geometric_augment = use_geometric_augment
        self.pregenerate = pregenerate

        # Validate image shape
        H, W = image_shape
        assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be divisible by 8"

        # Validate that either load_from_file or generate_fn is provided
        if load_from_file is None and generate_fn is None:
            # raise ValueError("Either 'load_from_file' or 'generate_fn' must be provided")
            generate_fn = generate_synthetic_image
            # Set default kwargs for generate_synthetic_image if not provided
            if generate_kwargs is None:
                # Default: generate grayscale images for training
                generate_kwargs = {'width': W, 'height': H, 'grayscale': True}

        # Set generate_fn and kwargs AFTER default assignment
        self.generate_fn = generate_fn
        self.generate_kwargs = generate_kwargs or {}

        # Ensure grayscale=True is set if not specified (for training efficiency)
        if 'grayscale' not in self.generate_kwargs and generate_fn == generate_synthetic_image:
            self.generate_kwargs['grayscale'] = True

        # Remove use_homography from generate_kwargs if using augmentation
        if self.use_homography_augment and 'use_homography' in self.generate_kwargs:
            print("⚠️  Removing 'use_homography' from generate_kwargs (will apply as augmentation)")
            self.generate_kwargs = {k: v for k, v in self.generate_kwargs.items()
                                    if k != 'use_homography'}

        # Pre-generate or load base samples
        self.pregenerated_data = None
        if load_from_file:
            print(f"Loading {num_samples} samples from {load_from_file}...")
            self._load_samples(load_from_file)
            print(f"✓ Loaded {len(self.pregenerated_data)} samples!")
        elif self.pregenerate:
            print(f"Pre-generating {num_samples} base samples...")
            self._pregenerate_samples()
            print(f"✓ Pre-generation complete!")

    def _pregenerate_samples(self):
        """Pre-generate base samples WITHOUT any augmentation."""
        self.pregenerated_data = []

        for i in range(self.num_samples):
            if (i + 1) % 500 == 0 or (i + 1) == self.num_samples:
                print(f"  {i + 1}/{self.num_samples} samples")

            # Generate base image and keypoints
            img, keypoints = self.generate_fn(**self.generate_kwargs)

            # Convert to grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img

            # Resize if needed
            H, W = self.image_shape
            if img_gray.shape != (H, W):
                img_gray = cv2.resize(img_gray, (W, H))

            # Store as uint8
            self.pregenerated_data.append((img_gray.copy(), keypoints.copy()))

    def _load_samples(self, filepath):
        """Load pre-generated samples from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        images = data['images']
        keypoints_list = data['keypoints']

        self.pregenerated_data = []
        for i in range(min(self.num_samples, len(images))):
            self.pregenerated_data.append((images[i], keypoints_list[i]))

    def save_to_file(self, filepath):
        """Save pre-generated samples to .npz file."""
        if self.pregenerated_data is None:
            raise ValueError("No pre-generated data to save. Run with pregenerate=True first.")

        images = np.array([item[0] for item in self.pregenerated_data], dtype=np.uint8)
        keypoints_list = np.array([item[1] for item in self.pregenerated_data], dtype=object)

        np.savez_compressed(filepath, images=images, keypoints=keypoints_list)
        print(f"✓ Saved {len(self.pregenerated_data)} samples to {filepath}")

    def __len__(self):
        """Return actual number of pre-generated samples."""
        if self.pregenerated_data is not None:
            return len(self.pregenerated_data)
        else:
            return self.num_samples

    def __getitem__(self, idx):
        """
        Get training sample with augmentations.

        Returns:
            image: (1, H, W) tensor
            target: (65, H/8, W/8) tensor
        """
        # Get base image and keypoints
        if self.pregenerated_data is not None:
            # Use pregenerated or loaded data
            img_gray, keypoints = self.pregenerated_data[idx]
            img_gray = img_gray.copy()
            keypoints = keypoints.copy()
        else:
            # Generate fresh sample on-the-fly
            img, keypoints = self.generate_fn(**self.generate_kwargs)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img

        # Apply augmentations in order:
        # 1. Homography (perspective transformation)
        if self.use_homography_augment:
            img_gray, keypoints = apply_homography_augmentation(
                img_gray, keypoints, self.image_shape
            )

        # 2. Photometric (brightness, contrast)
        if self.use_photometric_augment:
            img_gray = apply_photometric_augmentation(img_gray)

        # # 3. Geometric (flips)
        # if self.use_geometric_augment:
        #     img_gray, keypoints = apply_geometric_augmentation(
        #         img_gray, keypoints, self.image_shape
        #     )

        # Normalize to [0, 1]
        img_normalized = img_gray.astype(np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(img_normalized[np.newaxis, :, :])

        # Prepare training target
        target = prepare_training_target_basic([keypoints], self.image_shape)[0]

        return image_tensor, target
