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
    corners_pair = np.stack([src_corners, dst_corners], axis=-1).astype(np.float32)

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
    dst_corners, disp = perturb_corners(src_corners, disp_range)  # unpack the tuple

    # Ensure numeric arrays of shape (4,2) and dtype float32
    src = np.asarray(src_corners).reshape(-1, 2).astype(np.float32)
    dst = np.asarray(dst_corners).reshape(-1, 2).astype(np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(f"Expected corners shape (4,2), got src={src.shape}, dst={dst.shape}")

    # Compute homography
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)

    # Warp image
    warped = cv2.warpPerspective(img, H_inv, (w, h), flags=cv2.INTER_LINEAR)

    # Extract patches
    orig_patch = img[y:y + window_size, x:x + window_size]
    warped_patch = warped[y:y + window_size, x:x + window_size]

    # Calculate offsets
    offsets = dst - src

    # Determine crop area around the bounding box with padding
    pad = 30  # padding around the box
    all_corners = np.vstack([src, dst])
    min_x = max(0, int(np.min(all_corners[:, 0])) - pad)
    max_x = min(w, int(np.max(all_corners[:, 0])) + pad)
    min_y = max(0, int(np.min(all_corners[:, 1])) - pad)
    max_y = min(h, int(np.max(all_corners[:, 1])) + pad)

    # Crop images
    img_crop = img[min_y:max_y, min_x:max_x]
    warped_crop = warped[min_y:max_y, min_x:max_x]

    # Adjust corner coordinates for cropped images
    src_crop = src - np.array([min_x, min_y])
    dst_crop = dst - np.array([min_x, min_y])
    x_crop = x - min_x
    y_crop = y - min_y

    # Plot - single row with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1. Original image (cropped) with both source and destination corners
    ax = axes[0]
    ax.imshow(img_crop, cmap='gray')
    # Draw green box for source corners
    rect_src = plt.Rectangle((x_crop, y_crop), window_size, window_size, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect_src)
    # Draw source corners
    for i, (cx, cy) in enumerate(src_crop):
        ax.plot(cx, cy, 'go', markersize=8)
        ax.text(cx, cy - 3, f'{i}', color='green', fontsize=9, ha='center')

    # Draw red polygon for destination corners
    dst_polygon = plt.Polygon(dst_crop, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(dst_polygon)
    # Draw destination corners
    for i, (cx, cy) in enumerate(dst_crop):
        ax.plot(cx, cy, 'ro', markersize=8)
        ax.text(cx, cy - 3, f'{i}', color='red', fontsize=9, ha='center')

    # Draw arrows showing displacement
    for i in range(4):
        ax.arrow(src_crop[i, 0], src_crop[i, 1],
                 offsets[i, 0], offsets[i, 1],
                 head_width=2, head_length=2, fc='yellow', ec='yellow', alpha=0.6)

    ax.set_title(f'Original (Green→Red)\nAvg offset: {np.abs(offsets).mean():.1f}px', fontsize=10)
    ax.axis('off')

    # 2. Warped image (cropped) with H^-1
    ax = axes[1]
    ax.imshow(warped_crop, cmap='gray')
    rect = plt.Rectangle((x_crop, y_crop), window_size, window_size, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    ax.set_title('Warped (H⁻¹)', fontsize=10)
    ax.axis('off')

    # 3. Original patch
    ax = axes[2]
    ax.imshow(orig_patch, cmap='gray')
    ax.set_title('Original Patch', fontsize=10)
    ax.axis('off')

    # 4. Warped patch
    ax = axes[3]
    ax.imshow(warped_patch, cmap='gray')
    ax.set_title('Warped Patch', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


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


def get_random_images(num_images, image_dir="datasets/val2017_preprocessed"):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

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
