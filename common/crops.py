import cv2
import numpy as np

from common.config import DIGIT_COUNT

# Leave room for faint strokes beyond the threshold-derived boxes.
CROP_MARGIN = 0.08
# Start with enough resolution to inspect thin strokes; model size is undecided.
CROP_SIDE = 64
# Keep a visible border after aspect-preserving resizing.
INNER_SIDE = 56


def border_intensity(patch):
    """Estimate the local background for rotation padding from the patch edges."""
    border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1]))
    return int(np.median(border))


def padded_patch(gray, boxes, index):
    """Add margins without crossing image edges or expanding into neighboring digits."""
    x, y, width, height = boxes[index]
    margin = max(2, round(height * CROP_MARGIN))
    left, right = x - margin, x + width + margin
    if index:
        previous = boxes[index - 1]
        left = max(left, min(x, (previous[0] + previous[2] + x) // 2))
    if index + 1 < len(boxes):
        right = min(right, max(x + width, (x + width + boxes[index + 1][0]) // 2))
    left, right = max(0, left), min(gray.shape[1], right)
    top, bottom = max(0, y - margin), min(gray.shape[0], y + height + margin)
    if width <= 0 or height <= 0 or right <= left or bottom <= top:
        raise ValueError('Digit box does not intersect the image')
    return gray[top:bottom, left:right]


def straighten_patch(patch, angle):
    """The localizer's downward-positive slope needs the same OpenCV rotation sign."""
    height, width = patch.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    target_width = int(np.ceil(width * cosine + height * sine))
    target_height = int(np.ceil(height * cosine + width * sine))
    matrix[:, 2] += [(target_width - width) / 2, (target_height - height) / 2]
    background = border_intensity(patch)
    rotated = cv2.warpAffine(patch, matrix, (target_width, target_height),
                             flags=cv2.INTER_LINEAR, borderValue=background)
    return rotated, background


def prepare_crops(gray, localization):
    """Prepare seven ordered grayscale crops with deskewing and aspect-preserving padding."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Expected a nonempty uint8 grayscale image')
    if len(localization.digit_boxes) != DIGIT_COUNT:
        raise ValueError('Expected seven digit boxes')
    if localization.polarity not in ('light', 'dark') or not np.isfinite(localization.angle):
        raise ValueError('Invalid localization polarity or angle')
    crops = []
    for index in range(len(localization.digit_boxes)):
        patch = padded_patch(gray, localization.digit_boxes, index)
        patch, background = straighten_patch(patch, localization.angle)
        if localization.polarity == 'light':
            patch, background = 255 - patch, 255 - background
        scale = INNER_SIDE / max(patch.shape)
        size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(patch, size, interpolation=interpolation)
        canvas = np.full((CROP_SIDE, CROP_SIDE), background, dtype=np.uint8)
        left, top = (CROP_SIDE - size[0]) // 2, (CROP_SIDE - size[1]) // 2
        canvas[top:top + size[1], left:left + size[0]] = resized
        crops.append(canvas)
    return np.stack(crops)
