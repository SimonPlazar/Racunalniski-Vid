import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

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


def prepare_training_target_basic(keypoints_list, image_shape):
    """
    Priprava učnih ciljev iz seznama ključnih točk - osnovni način.

    Args:
        keypoints_list: list of (N, 2) numpy arrays z (x, y) koordinatami za vsak primer
        image_shape: (H, W) velikost slike

    Returns:
        target: (B, 65, H/8, W/8) ciljni tenzor za učenje
    """
    H, W = image_shape
    assert H % 8 == 0 and W % 8 == 0, "Velikost slike mora biti deljiva z 8"

    B = len(keypoints_list)
    Hc, Wc = H // 8, W // 8

    target = np.zeros((B, 65, Hc, Wc), dtype=np.float32)

    for b, keypoints in enumerate(keypoints_list):
        # 1. Ustvari masko s točkami
        mask = np.zeros((H, W), dtype=np.float32)

        if len(keypoints) > 0:
            # Zaokrožimo koordinate
            xs = np.clip(np.round(keypoints[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(keypoints[:, 1]).astype(int), 0, H - 1)
            mask[ys, xs] = 1.0

        # 2. Reshape v (H/8, 8, W/8, 8)
        mask_reshaped = mask.reshape(Hc, 8, Wc, 8)

        # 3. Permute v (H/8, W/8, 8, 8)
        mask_permuted = np.transpose(mask_reshaped, (0, 2, 1, 3))

        # 4. Reshape v (H/8, W/8, 64)
        mask_flat = mask_permuted.reshape(Hc, Wc, 64)

        # 5. Obdelava konfliktov - za vsak cell (H/8, W/8)
        for i in range(Hc):
            for j in range(Wc):
                cell = mask_flat[i, j]  # (64,)
                active_indices = np.where(cell > 0)[0]

                if len(active_indices) > 1:
                    # Več točk v istem cellu - ohrani naključno eno
                    keep_idx = np.random.choice(active_indices)
                    cell[:] = 0
                    cell[keep_idx] = 1.0
                elif len(active_indices) == 1:
                    # Ena točka - vse OK
                    pass
                else:
                    # Nobene točke - aktiviraj kanal 65 (kanal "ni točke")
                    pass  # pustimo vse na 0, kanal 65 dodamo spodaj

        # 6. Transponiraj v (64, H/8, W/8)
        target[b, :64, :, :] = np.transpose(mask_flat, (2, 0, 1))

        # 7. Dodaj kanal 65 - kjer nobeden od 64 kanalov ni aktiven
        no_point = (target[b, :64, :, :].sum(axis=0) == 0).astype(np.float32)
        target[b, 64, :, :] = no_point

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

    # Random horizontal flip
    if np.random.rand() < 0.5:
    # Random horizontal flip (50% chance)
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, 0] = W - 1 - keypoints[:, 0]

    # Random vertical flip
    if np.random.rand() < 0.5:
    # Random vertical flip (50% chance)
        if len(keypoints) > 0:
            keypoints = keypoints.copy()
            keypoints[:, 1] = H - 1 - keypoints[:, 1]

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
        self.generate_fn = generate_fn
        self.generate_kwargs = generate_kwargs or {}
        self.use_homography_augment = use_homography_augment
        self.use_photometric_augment = use_photometric_augment
        self.use_geometric_augment = use_geometric_augment
        self.pregenerate = pregenerate

        # Validate image shape
        H, W = image_shape
        assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be divisible by 8"

        # Validate that either load_from_file or generate_fn is provided
        if load_from_file is None and generate_fn is None:
            raise ValueError("Either 'load_from_file' or 'generate_fn' must be provided")

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

        # 3. Geometric (flips)
        if self.use_geometric_augment:
            img_gray, keypoints = apply_geometric_augmentation(
                img_gray, keypoints, self.image_shape
            )

        # Normalize to [0, 1]
        img_normalized = img_gray.astype(np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(img_normalized[np.newaxis, :, :])

        # Prepare training target
        target = prepare_training_target_basic([keypoints], self.image_shape)[0]

        return image_tensor, target

