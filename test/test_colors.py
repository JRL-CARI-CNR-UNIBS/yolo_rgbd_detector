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

"""Unit tests for yolo_rgbd_detector.colors."""

import numpy as np
import pytest

from yolo_rgbd_detector.colors import (
    classify_pixels,
    COLOR_NAMES,
    dominant_color,
    object_color,
)


def rgb(r, g, b):
    """Return a single BGR pixel from RGB components."""
    return np.array([[b, g, r]], dtype=np.uint8)


@pytest.mark.parametrize('pixel, expected', [
    ((0, 0, 0), 'black'),
    ((128, 128, 128), 'gray'),
    ((250, 250, 245), 'white'),
    ((220, 20, 20), 'red'),
    ((250, 140, 0), 'orange'),
    ((120, 70, 30), 'brown'),
    ((240, 230, 30), 'yellow'),
    ((30, 180, 40), 'green'),
    ((20, 200, 220), 'cyan'),
    ((20, 60, 200), 'blue'),
    ((130, 40, 200), 'purple'),
    ((240, 80, 180), 'pink'),
])
def test_classify_pixels(pixel, expected):
    label = classify_pixels(rgb(*pixel))[0]
    assert COLOR_NAMES[label] == expected


def test_dominant_color_majority():
    pixels = np.concatenate([np.repeat(rgb(20, 60, 200), 70, axis=0),
                             np.repeat(rgb(250, 250, 250), 30, axis=0)])
    est = dominant_color(pixels)
    assert est.name == 'blue'
    assert est.fraction == pytest.approx(0.7)
    assert est.rgb == pytest.approx((20 / 255, 60 / 255, 200 / 255))
    assert dominant_color(np.zeros((0, 3), np.uint8)) is None


def test_object_color_mask_vs_box():
    image = np.full((100, 100, 3), 255, np.uint8)       # white background
    image[40:60, 40:60] = (0, 0, 220)                   # small red square (BGR)
    box = (20, 20, 80, 80)
    # Central 50% of the box (30..70) is 400 red px out of 1600: white wins.
    assert object_color(image, box, roi_scale=0.5).name == 'white'
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    assert object_color(image, box, mask=mask).name == 'red'
    assert object_color(image, (200, 200, 300, 300)) is None


def test_object_color_subsampling():
    image = np.zeros((400, 400, 3), np.uint8)
    image[:, :] = (200, 60, 20)  # blue (BGR)
    est = object_color(image, (0, 0, 400, 400), roi_scale=1.0, max_pixels=1000)
    assert est.name == 'blue' and est.fraction == 1.0
