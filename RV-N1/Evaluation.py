import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from Generator import generate_pair
from Models import classes_to_offsets


# ============================================================
# GENERATE TEST SET
# ============================================================

def generate_test_set(images, samples_per_image=10, window_size=64, margin=16,
                     disp_range=(-16, 16), seed=None):
    """
    Generate a test set from a list of images.

    Args:
        images: List of grayscale images
        samples_per_image: Number of samples to generate per image
        window_size: Size of the patch window
        margin: Margin from image edges
        disp_range: Range for random displacements
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries, each containing:
            - 'pair': (H, W, 2) stacked patch pair
            - 'offsets': (4, 2) ground truth offsets
            - 'src_corners': (4, 2) source corners
            - 'dst_corners': (4, 2) destination corners
            - 'image_idx': index of source image
    """
    if seed is not None:
        np.random.seed(seed)

    n_images = len(images)
    test_samples = []

    for img_idx in tqdm(range(n_images), desc="Generating test set"):
        img = images[img_idx]
        for _ in range(samples_per_image):
            pair, offsets, corners_pair, image_pair = generate_pair(
                img, window_size=window_size, margin=margin, disp_range=disp_range
            )

            # Extract src and dst corners from corners_pair
            # corners_pair shape: (4, 2, 2) where [:,:,0] is src, [:,:,1] is dst
            if corners_pair.ndim == 3 and corners_pair.shape[2] == 2:
                src_corners = corners_pair[:, :, 0]
                dst_corners = corners_pair[:, :, 1]
            else:
                # Fallback: compute from offsets
                src_corners = corners_pair.reshape(4, 2)
                dst_corners = src_corners + offsets.reshape(4, 2)

            test_samples.append({
                'pair': pair,  # (64, 64, 2)
                'offsets': offsets.reshape(4, 2),  # (4, 2)
                'src_corners': src_corners,  # (4, 2)
                'dst_corners': dst_corners,  # (4, 2)
                'image_idx': img_idx
            })

    return test_samples


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_classical(test_samples):
    """
    Evaluate classical homography estimation using SIFT feature matching.

    Args:
        test_samples: List of test samples from generate_test_set

    Returns:
        Dictionary with metrics and predictions
    """
    predictions = []
    corner_errors = []
    rmse_values = []
    mae_values = []
    failed_count = 0

    # Initialize SIFT detector
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    for sample in tqdm(test_samples, desc="Evaluating classical (SIFT)"):
        src_corners = sample['src_corners']
        dst_corners = sample['dst_corners']
        pair = sample['pair']  # (64, 64, 2)

        # Extract patches
        patch1 = (pair[:, :, 0] * 255).astype(np.uint8)  # Source patch
        patch2 = (pair[:, :, 1] * 255).astype(np.uint8)  # Warped patch

        H = None

        # Use SIFT feature matching
        try:
            kp1, des1 = detector.detectAndCompute(patch1, None)
            kp2, des2 = detector.detectAndCompute(patch2, None)

            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                # Match features
                matches = matcher.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                # Need at least 4 matches for homography
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except Exception:
            # Handle any errors during feature matching
            pass

        # Compute predicted corners
        if H is not None:
            try:
                # Define the 4 corners of the source patch in patch coordinates
                h, w = patch1.shape
                patch_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

                # Apply homography
                patch_corners_h = np.concatenate([patch_corners, np.ones((4, 1))], axis=1)
                pred_corners_h = (H @ patch_corners_h.T).T

                # Check for valid homogeneous coordinates (avoid division by zero)
                z_coords = pred_corners_h[:, 2:3]
                if np.any(np.abs(z_coords) < 1e-8):
                    # Degenerate homography (division by near-zero)
                    raise ValueError("Degenerate homography")

                pred_patch_corners = pred_corners_h[:, :2] / z_coords

                # Check for NaN or Inf values
                if np.any(~np.isfinite(pred_patch_corners)):
                    raise ValueError("Invalid corner coordinates")

                # Compute offsets in patch coordinates
                pred_offsets = pred_patch_corners - patch_corners

                # Additional sanity check: offsets shouldn't be too extreme
                if np.any(np.abs(pred_offsets) > 200):
                    raise ValueError("Extreme offset values detected")

            except Exception:
                # Handle any errors during homography application
                failed_count += 1
                pred_offsets = np.zeros((4, 2))
                H = None  # Mark as failed
        else:
            # Homography estimation failed
            failed_count += 1
            pred_offsets = np.zeros((4, 2))

        # Ground truth offsets
        gt_offsets = sample['offsets']  # (4, 2)

        # Compute errors
        error = pred_offsets - gt_offsets
        corner_errors.append(np.linalg.norm(error, axis=1))  # (4,) L2 error per corner
        rmse_values.append(np.sqrt(np.mean(error ** 2)))
        mae_values.append(np.mean(np.abs(error)))

        predictions.append({
            'pred_offsets': pred_offsets,
            'gt_offsets': gt_offsets,
            'src_corners': src_corners,
            'pred_corners': src_corners + pred_offsets,
            'gt_corners': dst_corners,
            'H': H,
            'failed': H is None
        })

    if failed_count > 0:
        print(f"\n⚠️  Warning: {failed_count}/{len(test_samples)} ({failed_count/len(test_samples)*100:.1f}%) homographies failed")

    return {
        'predictions': predictions,
        'corner_errors': np.array(corner_errors),  # (N, 4)
        'rmse': np.array(rmse_values),  # (N,)
        'mae': np.array(mae_values),  # (N,)
        'mean_rmse': np.mean(rmse_values),
        'mean_mae': np.mean(mae_values),
        'mean_corner_error': np.mean(corner_errors),
        'failed_count': failed_count,
        'success_rate': (len(test_samples) - failed_count) / len(test_samples) * 100
    }


def evaluate_regressor(model, test_samples, device=None):
    """
    Evaluate a regression model.

    Args:
        model: HomographyRegressor model
        test_samples: List of test samples from generate_test_set
        device: torch device (auto-detected if None)

    Returns:
        Dictionary with metrics and predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    predictions = []
    corner_errors = []
    rmse_values = []
    mae_values = []

    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Evaluating regressor"):
            # Prepare input
            pair = sample['pair']  # (64, 64, 2)
            pair_tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Predict
            pred_offsets_flat = model(pair_tensor).cpu().numpy().flatten()  # (8,)
            pred_offsets = pred_offsets_flat.reshape(4, 2)

            # Ground truth
            gt_offsets = sample['offsets']  # (4, 2)
            src_corners = sample['src_corners']

            # Compute corners
            pred_corners = src_corners + pred_offsets
            gt_corners = sample['dst_corners']

            # Compute errors
            error = pred_offsets - gt_offsets
            corner_errors.append(np.linalg.norm(error, axis=1))  # (4,)
            rmse_values.append(np.sqrt(np.mean(error ** 2)))
            mae_values.append(np.mean(np.abs(error)))

            predictions.append({
                'pred_offsets': pred_offsets,
                'gt_offsets': gt_offsets,
                'src_corners': src_corners,
                'pred_corners': pred_corners,
                'gt_corners': gt_corners
            })

    return {
        'predictions': predictions,
        'corner_errors': np.array(corner_errors),  # (N, 4)
        'rmse': np.array(rmse_values),  # (N,)
        'mae': np.array(mae_values),  # (N,)
        'mean_rmse': np.mean(rmse_values),
        'mean_mae': np.mean(mae_values),
        'mean_corner_error': np.mean(corner_errors)
    }


def evaluate_classifier(model, test_samples, disp_range=(-16, 16), soft_decode=True, device=None):
    """
    Evaluate a classification model.

    Args:
        model: HomographyClassifier model
        test_samples: List of test samples from generate_test_set
        disp_range: Displacement range for class decoding
        soft_decode: Whether to use soft (weighted) or hard (argmax) decoding
        device: torch device (auto-detected if None)

    Returns:
        Dictionary with metrics and predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    predictions = []
    corner_errors = []
    rmse_values = []
    mae_values = []

    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Evaluating classifier"):
            # Prepare input
            pair = sample['pair']  # (64, 64, 2)
            pair_tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Predict
            logits = model(pair_tensor)  # (1, 8, 21)
            pred_offsets_tensor = classes_to_offsets(logits, disp_range, soft=soft_decode)
            pred_offsets = pred_offsets_tensor.cpu().numpy().reshape(4, 2)

            # Ground truth
            gt_offsets = sample['offsets']  # (4, 2)
            src_corners = sample['src_corners']

            # Compute corners
            pred_corners = src_corners + pred_offsets
            gt_corners = sample['dst_corners']

            # Compute errors
            error = pred_offsets - gt_offsets
            corner_errors.append(np.linalg.norm(error, axis=1))  # (4,)
            rmse_values.append(np.sqrt(np.mean(error ** 2)))
            mae_values.append(np.mean(np.abs(error)))

            predictions.append({
                'pred_offsets': pred_offsets,
                'gt_offsets': gt_offsets,
                'src_corners': src_corners,
                'pred_corners': pred_corners,
                'gt_corners': gt_corners
            })

    return {
        'predictions': predictions,
        'corner_errors': np.array(corner_errors),  # (N, 4)
        'rmse': np.array(rmse_values),  # (N,)
        'mae': np.array(mae_values),  # (N,)
        'mean_rmse': np.mean(rmse_values),
        'mean_mae': np.mean(mae_values),
        'mean_corner_error': np.mean(corner_errors),
        'soft_decode': soft_decode
    }


# ============================================================
# PLOTTING
# ============================================================

def summarize_and_plot(results_dict, save_dir=None, ylim=None):
    """
    Compare multiple models and plot results.

    Args:
        results_dict: Dictionary mapping model names to their evaluation results
                     Example: {
                         'Classical': classical_results,
                         'Regressor': regressor_results,
                         'Classifier (soft)': classifier_soft_results
                     }
        save_dir: Optional directory to save plots
        ylim: Y-axis limit for box plots. Can be:
              - None: auto (default)
              - float/int: set max limit (e.g., ylim=20)
              - tuple: set (min, max) (e.g., ylim=(0, 20))
    """
    n_models = len(results_dict)
    model_names = list(results_dict.keys())

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for name, results in results_dict.items():
        corner_errors_flat = results['corner_errors'].flatten()
        mean_error = np.mean(corner_errors_flat)
        std_error = np.std(corner_errors_flat)
        print(f"ocena z {name:<20} mean: {mean_error:>6.2f}  std: {std_error:>6.2f}")

    print("=" * 80 + "\n")

    # Create figure with subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for up to 5 models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # --- Plot 1: RMSE Distribution (Box Plot) ---
    ax = axes[0]
    rmse_data = [results['rmse'] for results in results_dict.values()]
    bp = ax.boxplot(rmse_data, labels=model_names, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors[:n_models]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('RMSE (pixels)', fontsize=12)
    ax.set_title('RMSE Distribution Across Models', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # Set y-axis limit if specified
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(0, ylim)
        elif isinstance(ylim, tuple) and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])

    # --- Plot 2: MAE Distribution (Box Plot) ---
    ax = axes[1]
    mae_data = [results['mae'] for results in results_dict.values()]
    bp = ax.boxplot(mae_data, labels=model_names, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors[:n_models]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('MAE (pixels)', fontsize=12)
    ax.set_title('MAE Distribution Across Models', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # Set y-axis limit if specified
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(0, ylim)
        elif isinstance(ylim, tuple) and len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])

    plt.tight_layout()

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_dir}/model_comparison.png")

    plt.show()

    # Additional statistics
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)

    for name, results in results_dict.items():
        print(f"\n{name}:")
        print(f"  RMSE - Mean: {results['mean_rmse']:.4f}, Std: {results['rmse'].std():.4f}")
        print(f"  MAE  - Mean: {results['mean_mae']:.4f}, Std: {results['mae'].std():.4f}")
        print(f"  Corner Errors - Mean: {results['mean_corner_error']:.4f}")

        # Percentiles
        corner_errors_flat = results['corner_errors'].flatten()
        p50 = np.percentile(corner_errors_flat, 50)
        p90 = np.percentile(corner_errors_flat, 90)
        p95 = np.percentile(corner_errors_flat, 95)
        print(f"  Percentiles - 50th: {p50:.4f}, 90th: {p90:.4f}, 95th: {p95:.4f}")

        # Accuracy at thresholds
        for thresh in [1, 2, 5]:
            acc = (corner_errors_flat < thresh).mean() * 100
            print(f"  Accuracy < {thresh}px: {acc:.2f}%")

    print("=" * 80 + "\n")

    return results_dict
