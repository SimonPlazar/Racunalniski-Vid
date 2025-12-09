import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter


# ============================================================================
# Utility Functions
# ============================================================================

def get_random_color(min_diff=50, exclude_color=None, grayscale=None):
    """Generate a random color or grayscale value.

    Args:
        min_diff: Minimum color difference from excluded color
        exclude_color: Color to avoid
        grayscale: Whether to generate grayscale (auto-detect if None)

    Returns:
        Color tuple (B,G,R) for RGB or (brightness,) for grayscale
    """
    # Auto-detect grayscale from exclude_color if not specified
    if grayscale is None:
        if exclude_color is not None:
            grayscale = len(exclude_color) == 1 if isinstance(exclude_color, tuple) else False
        else:
            grayscale = False

    if grayscale:
        # Generate single brightness value
        while True:
            brightness = random.randint(0, 255)
            if exclude_color is not None:
                exclude_val = exclude_color[0] if isinstance(exclude_color, tuple) else exclude_color
                diff = abs(brightness - exclude_val)
                if diff < min_diff:
                    continue
            return (brightness,)
    else:
        # Generate RGB color
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if exclude_color is not None:
                diff = sum(abs(a - b) for a, b in zip(color, exclude_color))
                if diff < min_diff:
                    continue
            return color


def add_gaussian_noise(image, sigma=None):
    """Add Gaussian noise to the image for realism."""
    if sigma is None:
        sigma = random.uniform(0.01, 0.05)

    # Add noise
    noisy = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, image.shape)
    noisy = noisy + noise
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)

    # Apply slight blur for smoothing
    blur_sigma = random.uniform(0.1, 0.3)

    if len(image.shape) == 2:
        noisy = gaussian_filter(noisy, sigma=blur_sigma)
    else:
        for i in range(3):
            noisy[:, :, i] = gaussian_filter(noisy[:, :, i], sigma=blur_sigma)

    return noisy


def check_overlap(mask1, mask2):
    """Check if two binary masks overlap."""
    return np.any(np.logical_and(mask1, mask2))


# ============================================================================
# Simple Shape Generators (triangles, quadrilaterals, stars)
# ============================================================================

def generate_triangle(width, height):
    """Generate a triangle shape with 3 keypoints."""
    margin = 30

    # Generate center point
    center_x = random.randint(margin + 40, width - margin - 40)
    center_y = random.randint(margin + 40, height - margin - 40)

    # Make sure triangle is at least 30% of image size
    min_size = min(width, height) * 0.3
    max_size = min(center_x - margin, width - center_x - margin,
                   center_y - margin, height - center_y - margin)

    # Generate 3 points around the center
    points = []
    for i in range(3):
        angle = (i * 2 * np.pi / 3) + random.uniform(-0.3, 0.3)
        radius = random.uniform(min_size, max_size)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    return points, "triangle"


def generate_quadrilateral(width, height):
    """Generate a quadrilateral shape with 4 keypoints."""
    margin = 30

    # Generate center point
    center_x = random.randint(margin + 40, width - margin - 40)
    center_y = random.randint(margin + 40, height - margin - 40)

    # Make sure quad is at least 30% of image size
    min_size = min(width, height) * 0.3
    max_size = min(center_x - margin, width - center_x - margin,
                   center_y - margin, height - center_y - margin)

    # Generate 4 points around the center
    points = []
    for i in range(4):
        angle = (i * 2 * np.pi / 4) + random.uniform(-0.2, 0.2)
        radius = random.uniform(min_size, max_size)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    return points, "quadrilateral"


def generate_star(width, height, num_points=5, min_angle_deg=30):
    """Generate a star shape with center + outer points as keypoints.

    Args:
        width, height: Image dimensions
        num_points: Number of star points (default 5)
        min_angle_deg: Minimum angle between points in degrees

    Returns:
        Array with first point = center, rest = outer vertices
    """
    # Keep margin so star doesn't touch edges
    margin = 40
    center_x = random.randint(margin + 10, max(margin + 10, width - margin - 10))
    center_y = random.randint(margin + 10, max(margin + 10, height - margin - 10))

    # Calculate radius range for outer points
    min_radius = int(max(5, min(width, height) * 0.12))
    max_radius_x = min(center_x - margin, width - margin - center_x)
    max_radius_y = min(center_y - margin, height - margin - center_y)
    max_radius = int(max(8, min(max_radius_x, max_radius_y)))

    # Convert to radians
    min_angle_rad = np.deg2rad(float(min_angle_deg))
    sector = 2 * np.pi / float(max(1, num_points))

    # Ensure minimum separation
    if min_angle_rad > sector * 0.9:
        min_angle_rad = sector * 0.5

    jitter_bound = max(0.0, (sector - min_angle_rad) / 2.0)

    # Generate angles with jitter
    angles = []
    for k in range(num_points):
        base = k * sector
        jitter = random.uniform(-jitter_bound, jitter_bound)
        angle = base + sector / 2.0 + jitter
        angles.append(angle)

    # Sort angles for consistent ordering
    angles = np.array(angles)
    angles = np.mod(angles, 2 * np.pi)
    angles.sort()

    # First point is the center
    points = [[float(center_x), float(center_y)]]

    # Add outer points
    for a in angles:
        r = min_radius if max_radius <= min_radius else random.randint(min_radius, max_radius)
        x = center_x + r * np.cos(a)
        y = center_y + r * np.sin(a)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    return points, "star"


def draw_simple_shape(img, points, color, shape_type):
    """Draw simple shapes on image."""
    if shape_type in ["triangle", "quadrilateral"]:
        cv2.fillPoly(img, [points.astype(np.int32)], color)
    elif shape_type == "star":
        center = points[0].astype(np.int32)
        for i in range(1, len(points)):
            pt = points[i].astype(np.int32)
            cv2.line(img, tuple(center), tuple(pt), color, 2, cv2.LINE_AA)


# ============================================================================
# Complex Shape Generators (checkerboard, cube)
# ============================================================================

def generate_checkerboard(width, height, rows=4, cols=5, randomize=True):
    """Generate checkerboard pattern with intersection points as keypoints.

    The board fills the image with a fixed 40px margin on all sides.
    Returns grid intersection points (rows+1 x cols+1).
    """
    # Ensure valid counts
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    # Fixed margin
    margin = 40

    # Calculate available space
    avail_w = max(1, width - 2 * margin)
    avail_h = max(1, height - 2 * margin)

    # Calculate cell size
    cell_w = avail_w / float(cols)
    cell_h = avail_h / float(rows)
    cell_size = int(max(2, np.floor(min(cell_w, cell_h))))

    # Calculate board dimensions and center it
    board_width = cell_size * cols
    board_height = cell_size * rows

    origin_x = margin + int((avail_w - board_width) / 2)
    origin_y = margin + int((avail_h - board_height) / 2)

    # Safety clamp
    origin_x = max(0, origin_x)
    origin_y = max(0, origin_y)

    # Generate intersection points (grid)
    keypoints = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = origin_x + j * cell_size
            y = origin_y + i * cell_size
            keypoints.append([float(x), float(y)])

    keypoints = np.array(keypoints, dtype=np.float32)

    params = {
        "rows": rows,
        "cols": cols,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_size": cell_size,
        "board_width": board_width,
        "board_height": board_height,
        "margin": margin
    }

    return keypoints, params, "checkerboard"


def draw_checkerboard(img, keypoints, params, color1, color2):
    """Draw checkerboard pattern with alternating colors."""
    rows = params["rows"]
    cols = params["cols"]
    origin_x = params["origin_x"]
    origin_y = params["origin_y"]
    cell_size = params["cell_size"]

    for i in range(rows):
        for j in range(cols):
            x1 = int(origin_x + j * cell_size)
            y1 = int(origin_y + i * cell_size)
            x2 = int(origin_x + (j + 1) * cell_size)
            y2 = int(origin_y + (i + 1) * cell_size)

            color = color1 if (i + j) % 2 == 0 else color2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


def generate_cube(width, height, randomize=True):
    """Generate cube geometry and two colors: cube fill and line color."""
    margin = 40
    cx = random.randint(margin + 20, max(margin + 20, width - margin - 20)) if randomize else width // 2
    cy = random.randint(margin + 20, max(margin + 20, height - margin - 20)) if randomize else height // 2

    min_dim = min(width, height)
    min_size = int(min_dim * 0.12)
    max_size = int(min_dim * 0.45)
    size = random.randint(max(24, min_size), max(32, max_size)) if randomize else int(min_dim * 0.28)

    half = size / 2.0
    # front square corners: TL, TR, BR, BL
    front = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]

    max_offset = size * 0.6
    offset_x = random.uniform(-max_offset, max_offset)
    offset_y = random.uniform(-max_offset, max_offset)
    if abs(offset_x) < 1.0 and abs(offset_y) < 1.0:
        offset_x = max_offset * 0.25
        offset_y = -max_offset * 0.25

    back = [(x + offset_x, y + offset_y) for (x, y) in front]

    front_np = np.array(front, dtype=np.float32)
    back_np = np.array(back, dtype=np.float32)

    # Determine visible back vertices (outside or on-edge of front polygon)
    padding = 2
    front_poly = front_np.astype(np.int32).reshape((-1, 1, 2))
    visible_back_idxs = []
    for idx, (x, y) in enumerate(back_np):
        dist = cv2.pointPolygonTest(front_poly, (float(x), float(y)), True)
        if dist <= -padding:
            visible_back_idxs.append(idx)

    # Ensure we get at least 1 and at most 3 visible back vertices
    if len(visible_back_idxs) < 1:
        # Fallback: if no vertices are visible, take the one furthest outside
        distances = [cv2.pointPolygonTest(front_poly, (float(x), float(y)), True)
                     for x, y in back_np]
        visible_back_idxs = [int(np.argmin(distances))]

    visible_back_idxs = visible_back_idxs[:3]

    # keypoints: 4 front corners then visible back corners
    keypoints_list = [list(pt) for pt in front_np]
    for idx in visible_back_idxs:
        keypoints_list.append(list(back_np[idx]))
    keypoints = np.array(keypoints_list, dtype=np.float32)

    params = {
        "front_full": front_np,
        "back_full": back_np,
        "visible_back_order": visible_back_idxs,
        "offset": (float(offset_x), float(offset_y))
    }

    return keypoints, params, "cube"


def draw_cube(img, keypoints, params, cube_color, line_color):
    """Draw cube given params and colors."""
    front = params["front_full"].astype(np.int32)
    back = params["back_full"].astype(np.int32)

    # Prepare front polygon for visibility tests
    front_poly = front.reshape((-1, 1, 2))

    # Determine whether each back vertex is outside/on-edge of the front polygon
    back_outside = []
    for k in range(4):
        pt = tuple(back[k].tolist())
        inside = cv2.pointPolygonTest(front_poly, pt, False)
        back_outside.append(inside <= 0)  # True == visible (outside or on edge)

    # Define faces by indices (front_i, front_j, back_j, back_i)
    faces = []
    visible_face_flags = []
    for i in range(4):
        j = (i + 1) % 4
        face = np.array([front[i], front[j], back[j], back[i]], dtype=np.int32)
        faces.append(face)
        # face is visible if either back vertex for the face is outside
        visible = back_outside[i] or back_outside[j]
        visible_face_flags.append(visible)

    # Fill visible side faces
    for i, visible in enumerate(visible_face_flags):
        if not visible:
            continue
        try:
            cv2.fillPoly(img, [faces[i]], cube_color)
        except Exception:
            pass

    # Fill front face last so it appears on top
    try:
        cv2.fillPoly(img, [front], cube_color)
    except Exception:
        pass

    # Collect unique edges that are actually visible:
    # - front-front edges: always visible (front face is on top)
    # - front-back edges: visible if corresponding back vertex is outside
    # - back-back edges: visible if both back vertices are outside
    def edge_key(a, b):
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        return (a, b) if a <= b else (b, a)

    edges = set()

    # front-front edges
    for i in range(4):
        j = (i + 1) % 4
        edges.add(edge_key(front[i], front[j]))

    # side edges and back edges conditioned on back_outside flags
    for i in range(4):
        j = (i + 1) % 4
        # front_j <-> back_j
        if back_outside[j]:
            edges.add(edge_key(front[j], back[j]))
        # front_i <-> back_i
        if back_outside[i]:
            edges.add(edge_key(front[i], back[i]))
        # back_j <-> back_i (back-back) both must be outside
        if back_outside[i] and back_outside[j]:
            edges.add(edge_key(back[j], back[i]))

    # Draw unique edges with line_color
    thickness = 2
    for a, b in edges:
        pt1 = (int(a[0]), int(a[1]))
        pt2 = (int(b[0]), int(b[1]))
        cv2.line(img, pt1, pt2, line_color, thickness, cv2.LINE_AA)

    # Stronger outline for front face
    cv2.polylines(img, [front], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)


# ============================================================================
# Multi-Shape Generation
# ============================================================================

def generate_multiple_shapes(width, height, num_shapes=3):
    """Generate multiple non-overlapping shapes in one image."""
    shapes_data = []
    masks = []

    attempts = 0
    max_attempts = num_shapes * 10

    while len(shapes_data) < num_shapes and attempts < max_attempts:
        attempts += 1

        # Randomly choose shape type
        shape_type = random.choice(["triangle", "quadrilateral", "star"])

        if shape_type == "triangle":
            points, stype = generate_triangle(width, height)
        elif shape_type == "quadrilateral":
            points, stype = generate_quadrilateral(width, height)
        else:
            points, stype = generate_star(width, height, num_points=random.choice([4, 5]))

        # Create mask for this shape
        temp_img = np.zeros((height, width), dtype=np.uint8)
        if stype in ["triangle", "quadrilateral"]:
            cv2.fillPoly(temp_img, [points.astype(np.int32)], 255)
        elif stype == "star":
            center = points[0].astype(np.int32)
            for i in range(1, len(points)):
                pt = points[i].astype(np.int32)
                cv2.line(temp_img, tuple(center), tuple(pt), 255, 4)

        # Dilate to check for proximity
        kernel = np.ones((10, 10), np.uint8)
        temp_mask = cv2.dilate(temp_img, kernel, iterations=1) > 0

        # Check overlap with existing shapes
        overlap = False
        for existing_mask in masks:
            if check_overlap(temp_mask, existing_mask):
                overlap = True
                break

        if not overlap:
            shapes_data.append((points, stype))
            masks.append(temp_mask)

    return shapes_data


# ============================================================================
# Homography Transformations
# ============================================================================

def generate_random_homography(width, height, margin_ratio=0.2):
    """Generate a random homography (perspective) transformation."""
    margin = int(min(width, height) * margin_ratio)

    # Define 4 random points in the image (one in each quadrant)
    src_points = []

    # Top-left quadrant
    x = random.randint(margin, width // 2 - margin)
    y = random.randint(margin, height // 2 - margin)
    src_points.append([x, y])

    # Top-right quadrant
    x = random.randint(width // 2 + margin, width - margin)
    y = random.randint(margin, height // 2 - margin)
    src_points.append([x, y])

    # Bottom-right quadrant
    x = random.randint(width // 2 + margin, width - margin)
    y = random.randint(height // 2 + margin, height - margin)
    src_points.append([x, y])

    # Bottom-left quadrant
    x = random.randint(margin, width // 2 - margin)
    y = random.randint(height // 2 + margin, height - margin)
    src_points.append([x, y])

    src_points = np.array(src_points, dtype=np.float32)

    # Destination points are the corners (with optional rotation)
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Random rotation by 90 degrees
    rotation = random.choice([0, 1, 2, 3])
    dst_points = np.roll(dst_points, rotation, axis=0)

    # Calculate homography
    H = cv2.getPerspectiveTransform(src_points, dst_points)

    return H


def apply_homography(img, keypoints, H, width, height):
    """Apply homography to image and keypoints."""
    # Transform image
    img_warped = cv2.warpPerspective(img, H, (width, height))

    # Transform keypoints
    keypoints_homogeneous = np.hstack([keypoints, np.ones((len(keypoints), 1))])
    keypoints_transformed = (H @ keypoints_homogeneous.T).T
    keypoints_transformed = keypoints_transformed[:, :2] / keypoints_transformed[:, 2:3]

    # Filter out points outside image bounds
    valid_mask = (keypoints_transformed[:, 0] >= 0) & (keypoints_transformed[:, 0] < width) & \
                 (keypoints_transformed[:, 1] >= 0) & (keypoints_transformed[:, 1] < height)

    keypoints_valid = keypoints_transformed[valid_mask]

    return img_warped, keypoints_valid


def filter_keypoints_in_bounds(keypoints, width, height):
    """Return only keypoints inside [0, width) x [0, height)."""
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.size == 0:
        return keypoints.reshape(-1, 2)
    mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height)
    return keypoints[mask]


# ============================================================================
# Main Image Generation Function
# ============================================================================

def generate_synthetic_image(width=256, height=256, shape_type="random", use_homography=False, grayscale=False):
    """Generate a synthetic image with known keypoint locations.

    Parameters:
        width, height: Image dimensions (must be divisible by 8)
        shape_type: Type of shape to generate
                   Options: "triangle", "quadrilateral", "star", "checkerboard",
                           "cube", "multiple", "random"
        use_homography: Whether to apply random perspective transformation
        grayscale: Generate grayscale (True) or RGB (False) image

    Returns:
        img: Image as uint8 array (H,W) or (H,W,3)
        keypoints: Array of keypoint coordinates (N, 2)
    """
    # Ensure dimensions are divisible by 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Generate background color/brightness
    if grayscale:
        bg_color = (random.randint(0, 255),)  # Single-value tuple for grayscale
        img = np.full((height, width), bg_color[0], dtype=np.uint8)
    else:
        bg_color = get_random_color(grayscale=False)  # RGB tuple
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Initialize keypoints
    keypoints = None

    # Choose random shape type if needed
    if shape_type == "random":
        shape_type = random.choice(["triangle", "quadrilateral", "star",
                                    "checkerboard", "cube", "multiple"])

    # Generate shape and keypoints
    if shape_type == "triangle":
        keypoints, _ = generate_triangle(width, height)
        color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        draw_simple_shape(img, keypoints, color, "triangle")

    elif shape_type == "quadrilateral":
        keypoints, _ = generate_quadrilateral(width, height)
        color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        draw_simple_shape(img, keypoints, color, "quadrilateral")

    elif shape_type == "star":
        num_points = random.choice([4, 5, 6])
        keypoints, _ = generate_star(width, height, num_points)
        color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        draw_simple_shape(img, keypoints, color, "star")

    elif shape_type == "checkerboard":
        rows = random.choice([3, 4, 5])
        cols = random.choice([3, 4, 5])
        keypoints, params, _ = generate_checkerboard(width, height, rows, cols)
        # Generate two contrasting colors for checkerboard
        color1 = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        color2 = get_random_color(min_diff=100, exclude_color=color1, grayscale=grayscale)
        draw_checkerboard(img, keypoints, params, color1, color2)

    elif shape_type == "cube":
        keypoints, params, _ = generate_cube(width, height)
        # Generate two contrasting colors for cube
        cube_color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        line_color = get_random_color(min_diff=100, exclude_color=cube_color, grayscale=grayscale)
        draw_cube(img, keypoints, params, cube_color, line_color)

    elif shape_type == "multiple":
        shapes_data = generate_multiple_shapes(width, height, num_shapes=random.randint(2, 4))
        keypoints = []
        used_colors = [bg_color]

        for points, stype in shapes_data:
            # Ensure each shape has a different color from bg and other shapes
            color = get_random_color(min_diff=120, exclude_color=bg_color, grayscale=grayscale)
            # Try to make it different from previously used colors too
            attempts = 0
            while attempts < 5:
                conflict = False
                for used_color in used_colors:
                    if grayscale:
                        if abs(color[0] - used_color[0]) < 80:
                            conflict = True
                            break
                    else:
                        if sum(abs(a - b) for a, b in zip(color, used_color)) < 120:
                            conflict = True
                            break
                if not conflict:
                    break
                color = get_random_color(min_diff=120, exclude_color=bg_color, grayscale=grayscale)
                attempts += 1

            used_colors.append(color)
            draw_simple_shape(img, points, color, stype)
            keypoints.append(points)

        if len(keypoints) > 0:
            keypoints = np.vstack(keypoints)
        else:
            # Fallback: generate a triangle if nothing was created
            keypoints, _ = generate_triangle(width, height)
            color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
            draw_simple_shape(img, keypoints, color, "triangle")

    else:
        # Unknown shape - default to triangle
        keypoints, _ = generate_triangle(width, height)
        color = get_random_color(min_diff=100, exclude_color=bg_color, grayscale=grayscale)
        draw_simple_shape(img, keypoints, color, "triangle")

    # Apply homography transformation if requested
    if use_homography:
        max_retries = 20
        retries_count = 0

        while retries_count < max_retries:
            H = generate_random_homography(width, height)
            img_warped, keypoints_warped = apply_homography(img, keypoints, H, width, height)

            # Accept if we kept at least some keypoints
            if len(keypoints_warped) > 0:
                img, keypoints = img_warped, keypoints_warped
                break

            retries_count += 1

        # If all retries failed, keep original image

    # Filter out any keypoints outside image bounds
    keypoints = filter_keypoints_in_bounds(keypoints, width, height)

    # Add Gaussian noise for realism
    img = add_gaussian_noise(img)

    return img, keypoints





