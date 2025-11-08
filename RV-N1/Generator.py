import cv2
from tqdm import tqdm
import numpy as np
import random
import os


# ============================================================
# PRETVORI DATASET V SIVINSKE IN SPREMENI VELIKOST
# ============================================================

def prepair_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE):
    # Ustvari izhodni direktorij, če še ne obstaja
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pridobi vse .jpg slike
    image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg'))]

    print(f"Najdenih {len(image_paths)} slik za obdelavo...")

    for path in tqdm(image_paths, desc="Obdelujem slike"):
        # Preberi sliko
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Napaka pri branju slike: {path}")
            continue

        # Pretvori v sivinsko
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Spremeni velikost (320x240)
        resized = cv2.resize(gray, TARGET_SIZE)

        # Ustvari enako ime datoteke v izhodni mapi
        filename = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Shrani obdelano sliko
        cv2.imwrite(output_path, resized)

    print("✅ Vse slike so uspešno predobdelane in shranjene v:")
    print(f"   {OUTPUT_DIR}")


# ============================================================
# HELPER FUNKCIJE ZA GENERIRANJE PAROV
# ============================================================

def sample_window(img_shape, window_size=64, margin=16):
    h, w = img_shape[:2]
    x = random.randint(margin, w - margin - window_size)
    y = random.randint(margin, h - margin - window_size)
    return x, y


def get_corners(x, y, window_size=64):
    return np.array([
        [x, y],
        [x + window_size, y],
        [x + window_size, y + window_size],
        [x, y + window_size]
    ], dtype=np.float32)


def perturb_corners(corners, disp_range=(-16, 16)):
    min_disp, max_disp = disp_range
    disp = np.random.randint(min_disp, max_disp + 1, size=corners.shape).astype(np.float32)
    return corners + disp, disp


def generate_pair(img, window_size=64, margin=16, disp_range=(-16, 16)):
    h, w = img.shape[:2]

    x, y = sample_window((h, w), window_size, margin)

    src_corners = get_corners(x, y, window_size)
    dst_corners, disp = perturb_corners(src_corners, disp_range)
    # print("src_corners: ", src_corners)
    # print("dst_corners: ", dst_corners)
    # print("disp: ", disp)

    # Homografija H (src -> dst) in njen inverz
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    H_inv = np.linalg.inv(H)

    # Warp celotne slike z H^-1
    warped = cv2.warpPerspective(img, H_inv, (w, h), flags=cv2.INTER_LINEAR)

    # Izreži patche
    orig_patch = img[y:y + window_size, x:x + window_size]
    warped_patch = warped[y:y + window_size, x:x + window_size]

    # Stack v 2 kanala in normaliziraj
    pair = np.stack([orig_patch, warped_patch], axis=-1).astype(np.float32) / 255.0

    # Ground truth: pomiki kotičkov
    offsets = (dst_corners - src_corners).astype(np.float32)
    # print("offsets: ", offsets)

    image_pair = np.stack([img, warped], axis=-1)
    corners_pair = np.stack([src_corners, dst_corners], axis=-1)

    return pair, offsets, corners_pair, image_pair


# ============================================================
# TEST GENERIRANJA PARA
# ============================================================
import matplotlib.pyplot as plt


def visualize_generate_pair(image_dir):
    # Nalož naključno sliko
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_path = random.choice(image_paths)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape[:2]
    window_size = 64
    margin = 16
    disp_range = (-16, 16)

    # Sample window
    x, y = sample_window((h, w), window_size, margin)

    # Get corners
    src_corners = get_corners(x, y, window_size)
    dst_corners = perturb_corners(src_corners, disp_range)

    # Compute homography
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    H_inv = np.linalg.inv(H)

    # Warp image
    warped = cv2.warpPerspective(img, H_inv, (w, h), flags=cv2.INTER_LINEAR)

    # Extract patches
    orig_patch = img[y:y + window_size, x:x + window_size]
    warped_patch = warped[y:y + window_size, x:x + window_size]

    # Calculate offsets
    offsets = dst_corners - src_corners

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original image with source corners
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray')
    for i, (cx, cy) in enumerate(src_corners):
        ax.plot(cx, cy, 'go', markersize=10)
        ax.text(cx, cy - 5, f'{i}', color='green', fontsize=12, ha='center')
    rect = plt.Rectangle((x, y), window_size, window_size, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect)
    ax.set_title('Original Image + Source Corners (green)')
    ax.axis('off')

    # 2. Original image with destination corners
    ax = axes[0, 1]
    ax.imshow(img, cmap='gray')
    for i, (cx, cy) in enumerate(dst_corners):
        ax.plot(cx, cy, 'ro', markersize=10)
        ax.text(cx, cy - 5, f'{i}', color='red', fontsize=12, ha='center')
    # Draw lines showing displacement
    for i in range(4):
        ax.arrow(src_corners[i, 0], src_corners[i, 1],
                 offsets[i, 0], offsets[i, 1],
                 head_width=3, head_length=3, fc='yellow', ec='yellow', alpha=0.7)
    rect = plt.Rectangle((x, y), window_size, window_size, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_title(f'Perturbed Corners (red)\nAvg offset: {np.abs(offsets).mean():.1f}px')
    ax.axis('off')

    # 3. Warped image with H^-1
    ax = axes[0, 2]
    ax.imshow(warped, cmap='gray')
    rect = plt.Rectangle((x, y), window_size, window_size, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    ax.set_title('Warped Image (H⁻¹ applied)')
    ax.axis('off')

    # 4. Original patch
    ax = axes[1, 0]
    ax.imshow(orig_patch, cmap='gray')
    ax.set_title('Original Patch (64×64)')
    ax.axis('off')

    # 5. Warped patch
    ax = axes[1, 1]
    ax.imshow(warped_patch, cmap='gray')
    ax.set_title('Warped Patch (64×64)')
    ax.axis('off')

    # Hide the 6th subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Print offsets
    print("\nCorner offsets (Δx, Δy):")
    for i, (dx, dy) in enumerate(offsets):
        print(f"  Corner {i}: ({dx:+.1f}, {dy:+.1f}) px")


# ============================================================
# FUNCTIONS TO LOAD IMAGES FROM DISK
# ============================================================

def get_images_from_names(image_names, image_dir):
    images = []
    for filename in image_names:
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
        else:
            print(f"⚠️ Warning: Could not load {filename}")
    return images


def get_random_images(num_images=None, image_dir="datasets/val2017_preprocessed"):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if num_images is None:
        num_images = len(image_paths)
    selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
    images = []
    for path in selected_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"⚠️ Warning: Could not load image at {path}")
    return images

def get_all_images(image_dir="datasets/val2017_preprocessed"):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"⚠️ Warning: Could not load image at {path}")
    return images