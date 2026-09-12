from itertools import combinations

import numpy as np

from common.config import DIGIT_COUNT, MAX_ANGLE

# Digit heights may vary, but fragments and hardware should not define a row.
HEIGHT_RATIO_LIMITS = (0.65, 1.45)
# A quarter-height residual tolerates centroid shifts without admitting nearby hardware.
MAX_LINE_RESIDUAL = 0.25
# Matches the existing localizer's supported tilt range.
MAX_SLOPE = float(np.tan(np.radians(MAX_ANGLE)))


def aligned_rows(components, digit_count=DIGIT_COUNT):
    rows = {}
    for first, second in combinations(components, 2):
        horizontal_distance = second[4] - first[4]
        mean_height = (first[3] + second[3]) / 2
        if abs(horizontal_distance) < mean_height:
            continue
        slope = (second[5] - first[5]) / horizontal_distance
        if abs(slope) > MAX_SLOPE:
            continue
        aligned = []
        for component in components:
            height_ratio = component[3] / mean_height
            residual = abs(component[5] - first[5] - slope * (component[4] - first[4]))
            if HEIGHT_RATIO_LIMITS[0] < height_ratio < HEIGHT_RATIO_LIMITS[1]:
                if residual < MAX_LINE_RESIDUAL * mean_height:
                    aligned.append(component)
        aligned.sort(key=lambda component: component[4])
        for start in range(len(aligned) - digit_count + 1):
            row = aligned[start:start + digit_count]
            rows[tuple(tuple(component[:4]) for component in row)] = row
    return list(rows.values())
