from dataclasses import dataclass

import cv2
import numpy as np

from common.config import DIGIT_COUNT, MAX_ANGLE
from common.rows import HEIGHT_RATIO_LIMITS, aligned_rows
from common.thresholds import threshold_masks

# Half resolution preserves digit strokes while reducing thresholding work.
WORK_SCALE = 0.5
# Cover the observed digit sizes relative to the full seal image.
HEIGHT_FRACTIONS = (0.035, 0.20)
# Reject squat hardware and very thin fragments.
ASPECT_LIMITS = (0.15, 1.0)
# Require some foreground coverage without rejecting hollow digits.
MIN_FILL = 0.20
# Reject irregular rows that admitted background in the training audit; not a probability.
MAX_ROW_SCORE = 0.20


@dataclass(frozen=True)
class Localization:
    digit_boxes: tuple
    polarity: str
    angle: float
    score: float
    method: str = 'otsu'


def digit_components(mask):
    _, _, statistics, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for index, (x, y, width, height, area) in enumerate(statistics):
        if index == 0:
            continue
        valid_height = HEIGHT_FRACTIONS[0] * mask.shape[0] < height < HEIGHT_FRACTIONS[1] * mask.shape[0]
        if valid_height and ASPECT_LIMITS[0] < width / height < ASPECT_LIMITS[1]:
            if area > MIN_FILL * width * height:
                components.append((x, y, width, height, *centers[index]))
    return components


def row_geometry(row):
    heights = np.array([box[3] for box in row], dtype=float)
    height_ratios = heights / np.median(heights)
    if height_ratios.min() < HEIGHT_RATIO_LIMITS[0] or height_ratios.max() > HEIGHT_RATIO_LIMITS[1]:
        return float('inf'), 0.0
    centers = np.array([box[4:6] for box in row])
    gaps = np.diff(centers[:, 0])
    if np.any(gaps <= 0):
        return float('inf'), 0.0
    slope, intercept = np.polyfit(centers[:, 0], centers[:, 1], 1)
    residual = np.abs(centers[:, 1] - slope * centers[:, 0] - intercept).mean()
    score = heights.std() / heights.mean() + gaps.std() / gaps.mean() + residual / heights.mean()
    return float(score), float(np.degrees(np.arctan(slope)))


def candidate_rows(components):
    if len(components) < DIGIT_COUNT:
        return []
    ordered = sorted(components, key=lambda box: box[5])
    median_height = float(np.median([box[3] for box in ordered]))
    groups = [[ordered[0]]]
    for component in ordered[1:]:
        if component[5] - groups[-1][-1][5] < median_height / 2:
            groups[-1].append(component)
        else:
            groups.append([component])
    rows = []
    for group in groups:
        horizontal = sorted(group, key=lambda box: box[4])
        rows.extend(horizontal[start:start + DIGIT_COUNT] for start in range(len(horizontal) - DIGIT_COUNT + 1))
    return rows


def restore_boxes(row, source_shape, resized_shape):
    """Round extents outward so resizing odd dimensions cannot clip boundary pixels."""
    scale_x, scale_y = source_shape[1] / resized_shape[1], source_shape[0] / resized_shape[0]
    boxes = []
    for x, y, width, height, center_x, center_y in row:
        left, top = int(np.floor(x * scale_x)), int(np.floor(y * scale_y))
        right = min(source_shape[1], int(np.ceil((x + width) * scale_x)))
        bottom = min(source_shape[0], int(np.ceil((y + height) * scale_y)))
        boxes.append((left, top, right - left, bottom - top))
    return tuple(boxes)


def find_threshold_row(components_by_polarity, method, source_shape, resized_shape, use_alignment=False):
    best = None
    scale_ratio = (source_shape[0] / resized_shape[0]) / (source_shape[1] / resized_shape[1])
    for polarity, components in components_by_polarity:
        rows = aligned_rows(components, DIGIT_COUNT) if use_alignment else candidate_rows(components)
        for row in rows:
            score, angle = row_geometry(row)
            angle = float(np.degrees(np.arctan(np.tan(np.radians(angle)) * scale_ratio)))
            if not np.isfinite(score) or not np.isfinite(angle) or score >= MAX_ROW_SCORE or abs(angle) > MAX_ANGLE:
                continue
            if best is None or score < best.score:
                boxes = restore_boxes(row, source_shape, resized_shape)
                stage = f'{method}_aligned' if use_alignment else method
                best = Localization(boxes, polarity, angle, score, stage)
    return best


def find_digits(gray: np.ndarray) -> Localization | None:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.size == 0 or gray.dtype != np.uint8:
        raise ValueError('Expected a nonempty uint8 grayscale image')
    if min(gray.shape) < 2:
        return None
    small = cv2.resize(gray, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    for method in ('otsu', 'adaptive'):
        components = [(polarity, digit_components(mask)) for polarity, mask in threshold_masks(blurred, method)]
        baseline = find_threshold_row(components, method, gray.shape, small.shape)
        if baseline is not None:
            return baseline
    # Fast grouping already computed the adaptive components; reuse them for alignment.
    for method in ('adaptive', 'adaptive_local', 'adaptive_repair'):
        if method != 'adaptive':
            components = [(polarity, digit_components(mask)) for polarity, mask in threshold_masks(blurred, method)]
        recovered = find_threshold_row(components, method, gray.shape, small.shape, use_alignment=True)
        if recovered is not None:
            return recovered
    return None
