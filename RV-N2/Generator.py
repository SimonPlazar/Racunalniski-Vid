import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter


# Utility functions

def get_random_color(min_diff=50, exclude_color=None):
    """Generate random color, optionally ensuring it's different from exclude_color."""
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if exclude_color is not None:
            diff = sum(abs(a - b) for a, b in zip(color, exclude_color))
            if diff < min_diff:
                continue
        return color


def add_gaussian_noise(image, sigma=None):
    """Add Gaussian noise to image."""
    if sigma is None:
        sigma = random.uniform(0.01, 0.05)

    noisy = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, image.shape)
    noisy = noisy + noise
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)

    # Apply slight Gaussian blur for smoothing
    for i in range(3):
        noisy[:, :, i] = gaussian_filter(noisy[:, :, i], sigma=0.5)

    return noisy


def check_overlap(mask1, mask2):
    """Check if two masks overlap."""
    return np.any(np.logical_and(mask1, mask2))


# Simple shape generators (triangles, quadrilaterals, stars)

def generate_triangle(width, height):
    """Generate a triangle with 3 keypoints."""
    margin = 30
    # Generate center point
    center_x = random.randint(margin + 40, width - margin - 40)
    center_y = random.randint(margin + 40, height - margin - 40)

    # Generate triangle with minimum size
    min_size = min(width, height) * 0.3  # At least 30% of image dimension
    max_size = min(center_x - margin, width - center_x - margin,
                   center_y - margin, height - center_y - margin)

    points = []
    for i in range(3):
        angle = (i * 2 * np.pi / 3) + random.uniform(-0.3, 0.3)  # Roughly evenly spaced
        radius = random.uniform(min_size, max_size)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    color = get_random_color()

    return points, color, "triangle"


def generate_quadrilateral(width, height):
    """Generate a quadrilateral with 4 keypoints."""
    margin = 30
    # Generate center point
    center_x = random.randint(margin + 40, width - margin - 40)
    center_y = random.randint(margin + 40, height - margin - 40)

    # Generate quadrilateral with minimum size
    min_size = min(width, height) * 0.3  # At least 30% of image dimension
    max_size = min(center_x - margin, width - center_x - margin,
                   center_y - margin, height - center_y - margin)

    points = []
    for i in range(4):
        angle = (i * 2 * np.pi / 4) + random.uniform(-0.2, 0.2)  # Roughly evenly spaced
        radius = random.uniform(min_size, max_size)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    color = get_random_color()

    return points, color, "quadrilateral"


def generate_star(width, height, num_points=5, min_angle_deg=30):
    """Generate a star with center + outer points as keypoints.

    The star is constructed in polar coordinates around a center chosen first.
    `min_angle_deg` enforces a minimum angular separation between outer points by
    splitting the circle into equal sectors and jittering each sector center.

    Returns an array with the first point being the center and the following
    points being the outer vertices in order.
    """
    # Keep a safe margin so the star doesn't touch the image border
    margin = 40
    center_x = random.randint(margin + 10, max(margin + 10, width - margin - 10))
    center_y = random.randint(margin + 10, max(margin + 10, height - margin - 10))

    # Minimum and maximum radius for outer points
    min_radius = int(max(5, min(width, height) * 0.12))  # at least ~12% of min dim
    # maximum radius limited by distance to edges from center and margin
    max_radius_x = min(center_x - margin, width - margin - center_x)
    max_radius_y = min(center_y - margin, height - margin - center_y)
    max_radius = int(max(8, min(max_radius_x, max_radius_y)))

    # In deg -> rad
    min_angle_rad = np.deg2rad(float(min_angle_deg))

    # Sector width
    sector = 2 * np.pi / float(max(1, num_points))

    # If sector is smaller than required minimum separation, clamp min angle
    if min_angle_rad > sector * 0.9:
        # ensure there is at least some jitter space
        min_angle_rad = sector * 0.5

    # jitter bound so that two adjacent points remain separated at least min_angle_rad
    jitter_bound = max(0.0, (sector - min_angle_rad) / 2.0)

    angles = []
    for k in range(num_points):
        base = k * sector
        jitter = random.uniform(-jitter_bound, jitter_bound)
        angle = base + sector / 2.0 + jitter  # center of sector plus jitter
        angles.append(angle)

    # Sort angles to form a consistent polygon order (optional)
    angles = np.array(angles)
    angles = np.mod(angles, 2 * np.pi)
    angles.sort()

    points = [[float(center_x), float(center_y)]]  # center first

    for a in angles:
        # choose radius uniformly in allowed range
        if max_radius <= min_radius:
            r = min_radius
        else:
            r = random.randint(min_radius, max_radius)
        x = center_x + r * np.cos(a)
        y = center_y + r * np.sin(a)
        points.append([x, y])

    points = np.array(points, dtype=np.float32)
    color = get_random_color()

    return points, color, "star"


def draw_simple_shape(img, points, color, shape_type):
    """Draw simple shape (triangle, quad, star) on image."""
    if shape_type in ["triangle", "quadrilateral"]:
        cv2.fillPoly(img, [points.astype(np.int32)], color)
    elif shape_type == "star":
        center = points[0].astype(np.int32)
        for i in range(1, len(points)):
            pt = points[i].astype(np.int32)
            cv2.line(img, tuple(center), tuple(pt), color, 2, cv2.LINE_AA)


# Complex shape generators (checkerboard, cube)

def generate_checkerboard(width, height, rows=4, cols=5, randomize=True):
    """Generate checkerboard intersections (keypoints) so the board spans the image,
    with margins determined by the limiting axis (shorter side relative to rows/cols).

    - This version enforces a fixed margin of 40 pixels on all sides so the board
      always stays inside that safe area. The board then fills as much of the
      remaining area as possible and is centered inside the margin.
    """
    # Ensure sensible integer counts
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    # Fixed margin required by the user
    margin = 40

    # Compute available area inside margins
    avail_w = max(1, width - 2 * margin)
    avail_h = max(1, height - 2 * margin)

    # Compute per-cell size from the limiting axis so the board fills the available area
    cell_w = avail_w / float(cols)
    cell_h = avail_h / float(rows)
    cell_size = int(max(2, np.floor(min(cell_w, cell_h))))

    # Recompute board size and center it inside the margin area
    board_width = cell_size * cols
    board_height = cell_size * rows

    origin_x = margin + int((avail_w - board_width) / 2)
    origin_y = margin + int((avail_h - board_height) / 2)

    # Safety clamp origin to image
    origin_x = max(0, origin_x)
    origin_y = max(0, origin_y)

    # Generate intersection points (rows+1 x cols+1 grid)
    keypoints = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = origin_x + j * cell_size
            y = origin_y + i * cell_size
            keypoints.append([float(x), float(y)])

    keypoints = np.array(keypoints, dtype=np.float32)

    # colors
    color1 = get_random_color()
    color2 = get_random_color(min_diff=80, exclude_color=color1)

    params = {
        "color1": color1,
        "color2": color2,
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


def draw_checkerboard(img, keypoints, params):
    """Draw checkerboard given params (origin, cell_size, rows, cols)."""
    color1 = params["color1"]
    color2 = params["color2"]
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
    front_poly = front_np.astype(np.int32).reshape((-1, 1, 2))
    visible_back_idxs = []
    for idx, (x, y) in enumerate(back_np):
        inside = cv2.pointPolygonTest(front_poly, (float(x), float(y)), False)
        if inside <= 0:
            visible_back_idxs.append(idx)
    visible_back_idxs = visible_back_idxs[:3]

    # keypoints: 4 front corners then visible back corners
    keypoints_list = [list(pt) for pt in front_np]
    for idx in visible_back_idxs:
        keypoints_list.append(list(back_np[idx]))
    keypoints = np.array(keypoints_list, dtype=np.float32)

    cube_color = get_random_color(min_diff=60, exclude_color=None)
    line_color = get_random_color(min_diff=80, exclude_color=cube_color)

    params = {
        "cube_color": cube_color,
        "line_color": line_color,
        "front_full": front_np,
        "back_full": back_np,
        "visible_back_order": visible_back_idxs,
        "offset": (float(offset_x), float(offset_y))
    }

    return keypoints, params, "cube"


def draw_cube(img, keypoints, params):
    """Draw cube: fill visible faces with cube_color and draw only edges visible to camera using line_color."""
    cube_color = params["cube_color"]
    line_color = params["line_color"]

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

# Multi-shape generation with overlap detection

def generate_multiple_shapes(width, height, num_shapes=3):
    """Generate multiple non-overlapping simple shapes."""
    all_keypoints = []
    shapes_data = []
    masks = []

    attempts = 0
    max_attempts = num_shapes * 10

    while len(shapes_data) < num_shapes and attempts < max_attempts:
        attempts += 1

        # Randomly choose shape type
        shape_type = random.choice(["triangle", "quadrilateral", "star"])

        if shape_type == "triangle":
            points, color, stype = generate_triangle(width, height)
        elif shape_type == "quadrilateral":
            points, color, stype = generate_quadrilateral(width, height)
        else:
            points, color, stype = generate_star(width, height, num_points=random.choice([4, 5]))

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
            shapes_data.append((points, color, stype))
            masks.append(temp_mask)

    return shapes_data


# Homography functions

def generate_random_homography(width, height, margin_ratio=0.2):
    """Generate random homography transformation."""
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


# Main dataset generator

def generate_synthetic_image(width=256, height=256, shape_type="random", use_homography=False):
    """
    Generate a synthetic image with keypoints.

    Parameters:
    - width, height: Image dimensions (should be divisible by 8)
    - shape_type: "triangle", "quadrilateral", "star", "checkerboard", "cube", "multiple", "random"
    - use_homography: Whether to apply random homography transformation

    Returns:
    - img: RGB image
    - keypoints: Nx2 array of keypoint locations
    """
    # Ensure dimensions are divisible by 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Random background color
    bg_color = get_random_color()
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Choose random shape type if needed
    if shape_type == "random":
        shape_type = random.choice(["triangle", "quadrilateral", "star",
                                    "checkerboard", "cube", "multiple"])

    # Generate shape and keypoints
    if shape_type == "triangle":
        keypoints, color, _ = generate_triangle(width, height)
        color = get_random_color(min_diff=80, exclude_color=bg_color)
        draw_simple_shape(img, keypoints, color, "triangle")

    elif shape_type == "quadrilateral":
        keypoints, color, _ = generate_quadrilateral(width, height)
        color = get_random_color(min_diff=80, exclude_color=bg_color)
        draw_simple_shape(img, keypoints, color, "quadrilateral")

    elif shape_type == "star":
        num_points = random.choice([4, 5, 6])
        keypoints, color, _ = generate_star(width, height, num_points)
        color = get_random_color(min_diff=80, exclude_color=bg_color)
        draw_simple_shape(img, keypoints, color, "star")

    elif shape_type == "checkerboard":
        rows = random.choice([3, 4, 5])
        cols = random.choice([3, 4, 5])
        keypoints, params, _ = generate_checkerboard(width, height, rows, cols)
        draw_checkerboard(img, keypoints, params)

    elif shape_type == "cube":
        keypoints, params, _ = generate_cube(width, height)
        draw_cube(img, keypoints, params)

    elif shape_type == "multiple":
        shapes_data = generate_multiple_shapes(width, height, num_shapes=random.randint(2, 4))
        keypoints = []
        for points, color, stype in shapes_data:
            color = get_random_color(min_diff=120, exclude_color=bg_color)
            draw_simple_shape(img, points, color, stype)
            keypoints.append(points)
        keypoints = np.vstack(keypoints)

    # Apply homography if requested
    if use_homography:
        while True:
            max_retries = 20
            retries_count = 0

            H = generate_random_homography(width, height)
            img_warped, keypoints_warped = apply_homography(img, keypoints, H, width, height)
            if len(keypoints) > 0:
                img, keypoints = img_warped, keypoints_warped
                break

            retries_count += 1
            if retries_count >= max_retries:
                print("⚠️  Homography application failed to retain keypoints after multiple attempts. Keeping original image.")
                break

    # Keep only keypoints inside bounds
    keypoints = filter_keypoints_in_bounds(keypoints, width, height)

    # Add Gaussian noise
    img = add_gaussian_noise(img)

    return img, keypoints
