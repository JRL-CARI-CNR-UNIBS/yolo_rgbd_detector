# Copyright 2026 Manuel Beschi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dominant color of an object, as a human-readable name.

Each pixel of the object is classified into one of :data:`COLOR_NAMES` from its
HSV value, and the most frequent name wins. Voting per pixel is more robust
than averaging the pixels: a blue mug with a white inside is still "blue" as
long as most of its visible pixels are blue, while the average would be a
pale, meaningless blue-gray.

No ROS dependency.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from yolo_rgbd_detector.geometry import Box, clip_box, scale_box

COLOR_NAMES = (
    'black', 'gray', 'white',
    'red', 'orange', 'brown', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
)

# Hue ranges in degrees [start, end) for saturated colors. Red wraps around 0.
_HUE_RANGES = (
    ('red', 0.0, 15.0),
    ('orange', 15.0, 40.0),
    ('yellow', 40.0, 70.0),
    ('green', 70.0, 165.0),
    ('cyan', 165.0, 200.0),
    ('blue', 200.0, 260.0),
    ('purple', 260.0, 290.0),
    ('pink', 290.0, 345.0),
    ('red', 345.0, 360.0),
)


@dataclass(frozen=True)
class ColorThresholds:
    """Thresholds of the pixel classifier (values in [0, 1])."""

    black_value: float = 0.18
    """Pixels darker than this are black, whatever their hue."""
    gray_saturation: float = 0.22
    """Pixels less saturated than this are black, gray or white."""
    white_value: float = 0.75
    """Unsaturated pixels brighter than this are white, else gray."""
    brown_value: float = 0.6
    """Red/orange pixels darker than this are brown."""


@dataclass(frozen=True)
class ColorEstimate:
    """Dominant color of an object."""

    name: str
    rgb: Tuple[float, float, float]
    """Mean RGB of the pixels of the dominant color, components in [0, 1]."""
    fraction: float
    """Fraction of the sampled pixels that have the dominant color."""


def classify_pixels(bgr: np.ndarray,
                    thresholds: ColorThresholds = ColorThresholds()) -> np.ndarray:
    """
    Classify pixels into color names.

    :param bgr: N x 3 (or H x W x 3) uint8 array of BGR pixels.
    :return: Integer array with the same leading shape, indices into
        :data:`COLOR_NAMES`.
    """
    pixels = np.asarray(bgr, dtype=np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV_FULL).reshape(-1, 3).astype(np.float32)
    hue = hsv[:, 0] * (360.0 / 256.0)
    sat = hsv[:, 1] / 255.0
    val = hsv[:, 2] / 255.0

    t = thresholds
    idx = {name: i for i, name in enumerate(COLOR_NAMES)}
    labels = np.full(hue.shape, -1, dtype=np.int32)

    for name, start, end in _HUE_RANGES:
        labels[(hue >= start) & (hue < end)] = idx[name]
    # Dark red/orange looks brown.
    warm = (labels == idx['red']) | (labels == idx['orange'])
    labels[warm & (val < t.brown_value)] = idx['brown']
    # Achromatic colors override the hue.
    gray = sat < t.gray_saturation
    labels[gray] = idx['gray']
    labels[gray & (val >= t.white_value)] = idx['white']
    labels[val < t.black_value] = idx['black']

    return labels.reshape(np.asarray(bgr).shape[:-1])


def dominant_color(bgr_pixels: np.ndarray,
                   thresholds: ColorThresholds = ColorThresholds()) -> Optional[ColorEstimate]:
    """
    Return the most frequent color name of a set of pixels.

    :param bgr_pixels: N x 3 uint8 BGR pixels.
    :return: The estimate, or ``None`` if there are no pixels.
    """
    pixels = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 3)
    if pixels.shape[0] == 0:
        return None
    labels = classify_pixels(pixels, thresholds)
    counts = np.bincount(labels, minlength=len(COLOR_NAMES))
    best = int(np.argmax(counts))
    selected = pixels[labels == best].astype(np.float32).mean(axis=0) / 255.0
    b, g, r = (float(c) for c in selected)
    return ColorEstimate(name=COLOR_NAMES[best], rgb=(r, g, b),
                         fraction=float(counts[best]) / pixels.shape[0])


def object_color(image_bgr: np.ndarray, box: Box, mask: Optional[np.ndarray] = None,
                 roi_scale: float = 0.5, max_pixels: int = 5000,
                 thresholds: ColorThresholds = ColorThresholds()) -> Optional[ColorEstimate]:
    """
    Dominant color of a detected object.

    Pixels come from the mask when given, otherwise from the central
    ``roi_scale`` part of the box (same sampling as the depth median). At most
    ``max_pixels`` evenly spaced pixels are used, to bound the cost.

    :param image_bgr: H x W x 3 uint8 BGR image.
    :param box: ``(x1, y1, x2, y2)`` in pixels.
    :param mask: Optional H x W boolean mask of the object.
    """
    height, width = image_bgr.shape[:2]
    region = box if mask is not None else scale_box(box, roi_scale)
    x1, y1, x2, y2 = clip_box(region, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image_bgr[y1:y2, x1:x2]
    if mask is not None:
        pixels = patch[mask[y1:y2, x1:x2].astype(bool)]
    else:
        pixels = patch.reshape(-1, 3)
    if pixels.shape[0] > max_pixels:
        step = int(np.ceil(pixels.shape[0] / max_pixels))
        pixels = pixels[::step]
    return dominant_color(pixels, thresholds)
