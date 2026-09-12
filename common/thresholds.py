import cv2

# A 101-pixel neighborhood at half resolution spans roughly one digit height.
ADAPTIVE_BLOCK_SIZE = 101
# Opposite offsets isolate bright or dark ink from its local background.
ADAPTIVE_OFFSET = 8
# Smaller neighborhoods separate ink from nearby plate edges.
LOCAL_BLOCK_SIZE = 51
# A lower offset retains faint strokes before reconnecting short vertical breaks.
REPAIR_OFFSET = 4
# Digit fragments are mostly separated vertically; keep horizontal growth narrow.
REPAIR_KERNEL_SIZE = (3, 9)


def threshold_masks(blurred, method):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for polarity, flag in (('light', cv2.THRESH_BINARY), ('dark', cv2.THRESH_BINARY_INV)):
        if method == 'otsu':
            _, mask = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
        elif method in ('adaptive', 'adaptive_local', 'adaptive_repair'):
            block_size = LOCAL_BLOCK_SIZE if method == 'adaptive_local' else ADAPTIVE_BLOCK_SIZE
            magnitude = REPAIR_OFFSET if method == 'adaptive_repair' else ADAPTIVE_OFFSET
            offset = -magnitude if polarity == 'light' else magnitude
            mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         flag, block_size, offset)
        else:
            raise ValueError(f'Unknown threshold method: {method}')
        if method == 'adaptive_repair':
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, REPAIR_KERNEL_SIZE)
            yield polarity, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, repair_kernel)
        else:
            yield polarity, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
