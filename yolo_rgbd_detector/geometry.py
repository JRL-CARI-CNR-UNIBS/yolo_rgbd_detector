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
Geometry used to turn 2D detections into 3D objects. No ROS dependency.

Conventions
-----------
* Boxes are ``(x1, y1, x2, y2)`` in pixels, as returned by Ultralytics
  (``boxes.xyxy``). ``x2``/``y2`` are treated as exclusive when slicing.
* A *points image* is an H x W x 3 ``float32`` array with the 3D point seen
  by each pixel, in the camera optical frame (x right, y down, z forward),
  NaN where there is no measurement. It is built from a depth image
  (:func:`depth_to_points`), an organized point cloud, or an unorganized
  cloud projected into the image (:func:`project_points`).
* Quaternions are ``(x, y, z, w)``.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k: Sequence[float]) -> 'Intrinsics':
        """Build from the row-major 3x3 ``K`` matrix of ``sensor_msgs/CameraInfo``."""
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))

    def is_valid(self) -> bool:
        """Return True if the focal lengths are positive."""
        return self.fx > 0.0 and self.fy > 0.0


def depth_to_meters(depth: np.ndarray, encoding: str, depth_scale: float = 0.0) -> np.ndarray:
    """
    Convert a raw depth image to a ``float32`` image in meters.

    :param depth: Raw depth image as returned by ``cv_bridge`` with ``passthrough``.
    :param encoding: ROS image encoding (``16UC1``, ``mono16`` or ``32FC1``).
    :param depth_scale: Meters per raw unit. ``0`` selects the ROS convention:
        ``0.001`` (millimeters) for 16-bit images and ``1.0`` for float images.
    :raises ValueError: If the encoding is not a supported depth encoding.
    """
    if encoding in ('16UC1', 'mono16'):
        scale = depth_scale if depth_scale > 0.0 else 0.001
    elif encoding == '32FC1':
        scale = depth_scale if depth_scale > 0.0 else 1.0
    else:
        raise ValueError(f'Unsupported depth encoding: {encoding!r}')
    return depth.astype(np.float32) * np.float32(scale)


def depth_to_points(depth_m: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """
    Back-project a depth image (meters) to a points image.

    Pixels with zero, negative or non-finite depth become NaN.
    """
    h, w = depth_m.shape[:2]
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    z = depth_m.astype(np.float32)
    invalid = ~np.isfinite(z) | (z <= 0.0)
    z = np.where(invalid, np.nan, z)
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=-1)


def project_points(points: np.ndarray, intrinsics: Intrinsics,
                   height: int, width: int) -> np.ndarray:
    """
    Project 3D points (camera optical frame) into a points image.

    When several points fall in the same pixel, the closest one is kept
    (z-buffer). Points behind the camera are ignored.

    :param points: N x 3 array.
    :return: H x W x 3 points image (NaN where no point projects).
    """
    out = np.full((height, width, 3), np.nan, dtype=np.float32)
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0.0)]
    if p.shape[0] == 0:
        return out
    u = np.round(intrinsics.fx * p[:, 0] / p[:, 2] + intrinsics.cx).astype(np.int64)
    v = np.round(intrinsics.fy * p[:, 1] / p[:, 2] + intrinsics.cy).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, p = u[inside], v[inside], p[inside]
    if p.shape[0] == 0:
        return out
    pixel = v * width + u
    order = np.lexsort((p[:, 2], pixel))           # by pixel, then by depth
    pixel, p = pixel[order], p[order]
    _, first = np.unique(pixel, return_index=True)  # closest point of each pixel
    out.reshape(-1, 3)[pixel[first]] = p[first]
    return out


def deproject(u: float, v: float, z: float, intrinsics: Intrinsics) -> Vec3:
    """Back-project pixel ``(u, v)`` at depth ``z`` (meters) to a 3D point."""
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (float(x), float(y), float(z))


# ----------------------------------------------------------------------------
# 2D regions
# ----------------------------------------------------------------------------
def scale_box(box: Box, factor: float) -> Box:
    """Scale a box around its center (``factor`` < 1 shrinks it)."""
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * factor, 0.5 * (y2 - y1) * factor
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def clip_box(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Round a box to integer pixels and clip it to the image (end exclusive)."""
    x1, y1, x2, y2 = box
    ix1 = int(np.clip(np.floor(x1), 0, width))
    iy1 = int(np.clip(np.floor(y1), 0, height))
    ix2 = int(np.clip(np.ceil(x2), 0, width))
    iy2 = int(np.clip(np.ceil(y2), 0, height))
    return ix1, iy1, ix2, iy2


def polygon_to_mask(polygon: np.ndarray, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Rasterize a segmentation polygon into a boolean mask.

    :param polygon: N x 2 array of ``(x, y)`` pixel coordinates
        (Ultralytics ``masks.xy[i]``, in original image coordinates).
    :param shape: ``(height, width)`` of the output mask.
    :return: The mask, or ``None`` if the polygon is empty or degenerate.
    """
    if polygon is None or len(polygon) < 3:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    if not mask.any():
        return None
    return mask.astype(bool)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return the ``(u, v)`` centroid of a boolean mask, or ``None`` if it is empty."""
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return None
    # +0.5: pixel i covers [i, i+1), consistent with Ultralytics box coordinates.
    return (float(us.mean()) + 0.5, float(vs.mean()) + 0.5)


def region_points(points_img: np.ndarray, box: Box, mask: Optional[np.ndarray] = None,
                  roi_scale: float = 0.5) -> np.ndarray:
    """
    Return the valid 3D points of a detection.

    Pixels come from the mask (limited to the box) when given, otherwise from
    the box shrunk by ``roi_scale`` around its center.

    :return: N x 3 array (camera frame), without NaN.
    """
    height, width = points_img.shape[:2]
    region = box if mask is not None else scale_box(box, roi_scale)
    x1, y1, x2, y2 = clip_box(region, width, height)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.float32)
    patch = points_img[y1:y2, x1:x2]
    pts = patch[mask[y1:y2, x1:x2].astype(bool)] if mask is not None else patch.reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def region_mask(shape, box: Box, mask: Optional[np.ndarray] = None,
                roi_scale: float = 0.5) -> np.ndarray:
    """
    Return the pixels used for a detection as a full-size boolean mask.

    Same rule as :func:`region_points`: the mask limited to the box, or the
    box shrunk by ``roi_scale``.
    """
    height, width = shape[:2]
    out = np.zeros((height, width), dtype=bool)
    region = box if mask is not None else scale_box(box, roi_scale)
    x1, y1, x2, y2 = clip_box(region, width, height)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(bool) if mask is not None else True
    return out


def exclude_occluders(points_img: np.ndarray, masks: Sequence[np.ndarray],
                      min_gap: float = 0.03) -> List[np.ndarray]:
    """
    Remove from each mask the pixels of the detections in front of it.

    A mask often covers the objects resting on (or standing in front of) its
    object: a book mask includes the cup on it. Detection ``j`` occludes
    ``i`` when the median depth of ``j`` is smaller than that of ``i`` by
    more than ``min_gap``; its pixels are then removed from mask ``i``.

    :param masks: Full-size boolean masks (see :func:`region_mask`).
    :return: New masks, same order.
    """
    z = points_img[..., 2]
    medians = []
    for m in masks:
        values = z[m]
        values = values[np.isfinite(values)]
        medians.append(float(np.median(values)) if values.size else float('inf'))
    out = []
    for i, m in enumerate(masks):
        result = m.copy()
        for j, other in enumerate(masks):
            if j != i and medians[j] < medians[i] - min_gap:
                result &= ~other
        out.append(result)
    return out


# ----------------------------------------------------------------------------
# 3D points
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthStats:
    """Statistics of the depth (z) of a set of points (meters)."""

    median: float
    low: float
    high: float
    num_valid: int


def filter_object_points(points: np.ndarray, min_depth: float = 0.0,
                         max_depth: float = float('inf'),
                         min_band: float = 0.03) -> Tuple[np.ndarray, Optional[DepthStats]]:
    """
    Keep the points that belong to the object, dropping background and noise.

    1. Points with depth outside ``(min_depth, max_depth]`` are discarded.
    2. ``m`` = median depth: the object fills most of the region, so the
       median lies on it even with some background pixels.
    3. Points farther than ``band`` from ``m`` along z are discarded, where
       ``band = max(min_band, lateral extent of the points)``: an object is
       not much deeper than it is wide.

    :param points: N x 3 points in the camera optical frame.
    :return: ``(filtered points, stats of the input depth)``; stats is None if
        no point is in range.
    """
    p = points[(points[:, 2] > min_depth) & (points[:, 2] <= max_depth)]
    if p.shape[0] == 0:
        return p, None
    z = p[:, 2]
    median = float(np.median(z))
    near = p[np.abs(z - median) <= max(min_band, 0.5 * median)]
    x_lo, x_hi = np.percentile(near[:, 0], (5.0, 95.0))
    band = max(min_band, float(x_hi - x_lo))
    kept = p[np.abs(z - median) <= band]
    low, high = np.percentile(z, (10.0, 90.0))
    return kept, DepthStats(median, float(low), float(high), int(p.shape[0]))


def surface_centroid(points: np.ndarray) -> Vec3:
    """Coordinate-wise median of the points: a robust point on the visible surface."""
    m = np.median(points, axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def quaternion_matrix(q: Quat) -> np.ndarray:
    """3x3 rotation matrix of a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_points(points: np.ndarray, translation: Vec3, rotation: Quat) -> np.ndarray:
    """Apply a rigid transform (``p' = R p + t``) to N x 3 points."""
    r = quaternion_matrix(rotation)
    return (np.asarray(points, dtype=np.float64) @ r.T + np.asarray(translation)).astype(
        np.float32)


def yaw_to_quaternion(yaw: float) -> Quat:
    """Quaternion of a rotation about z."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


@dataclass(frozen=True)
class Box3D:
    """Box rotated about the z axis of its frame."""

    center: Vec3
    yaw: float
    size: Vec3

    @property
    def quaternion(self) -> Quat:
        """Orientation as a quaternion."""
        return yaw_to_quaternion(self.yaw)


def fit_box(points: np.ndarray, mode: str = 'yaw', low: float = 2.0, high: float = 98.0,
            min_size: float = 0.005) -> Box3D:
    """
    Fit a box enclosing the points, ignoring outliers.

    * z extent: ``low``/``high`` percentiles of z.
    * xy: points farther from the xy median than the ``high`` percentile are
      dropped; then the minimum-area rectangle (``mode='yaw'``) or the
      percentile axis-aligned rectangle (``mode='axis_aligned'``).

    The x side of the box is the longer one; yaw is in ``(-pi/2, pi/2]``.
    The z axis of the frame should point up (gravity) for a meaningful box.

    :param points: N x 3 points (N >= 1).
    """
    p = np.asarray(points, dtype=np.float64)
    z_lo, z_hi = np.percentile(p[:, 2], (low, high))
    xy = p[:, :2]
    if xy.shape[0] > 10:
        r = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
        xy = xy[r <= np.percentile(r, high)]

    if mode == 'axis_aligned':
        x_lo, x_hi = np.percentile(xy[:, 0], (low, high))
        y_lo, y_hi = np.percentile(xy[:, 1], (low, high))
        cx, cy, sx, sy, yaw = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, x_hi - x_lo, y_hi - y_lo, 0.0
    elif mode == 'yaw':
        (cx, cy), (sx, sy), angle = cv2.minAreaRect(xy.astype(np.float32))
        yaw = math.radians(angle)
    else:
        raise ValueError(f"Unknown box mode {mode!r} (use 'yaw' or 'axis_aligned')")

    if sy > sx:
        sx, sy, yaw = sy, sx, yaw + math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # wrap to [-pi/2, pi/2)
    if yaw <= -math.pi / 2 + 1e-9:
        yaw += math.pi
    return Box3D(
        center=(float(cx), float(cy), float((z_lo + z_hi) / 2)),
        yaw=float(yaw),
        size=(max(float(sx), min_size), max(float(sy), min_size),
              max(float(z_hi - z_lo), min_size)),
    )


def box_distance(a: Box3D, b: Box3D) -> float:
    """Distance between box centers."""
    return float(np.linalg.norm(np.subtract(a.center, b.center)))
