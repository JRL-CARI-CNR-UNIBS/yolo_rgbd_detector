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

"""Unit tests for geometry, cloud parsing and multi-view fusion (no YOLO)."""

import math

import numpy as np
import pytest
from sensor_msgs.msg import PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header

from yolo_rgbd_detector.cloud import cloud_to_arrays
from yolo_rgbd_detector.fusion import complete_3d, merge_views, ViewDetection
from yolo_rgbd_detector.geometry import (
    clip_box,
    deproject,
    depth_to_meters,
    depth_to_points,
    exclude_occluders,
    filter_object_points,
    fit_box,
    Intrinsics,
    mask_centroid,
    polygon_to_mask,
    project_points,
    quaternion_matrix,
    region_mask,
    region_points,
    scale_box,
    surface_centroid,
    transform_points,
)

INTR = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def scene(background=3.0, obj=1.0, box=(200, 100, 300, 200), shape=(480, 640)):
    """Depth image (meters) with a box-shaped object in front of a wall."""
    depth = np.full(shape, background, dtype=np.float32)
    x1, y1, x2, y2 = box
    depth[y1:y2, x1:x2] = obj
    return depth


def test_intrinsics_and_depth_conversion():
    intr = Intrinsics.from_k([600.0, 0.0, 321.0, 0.0, 601.0, 239.0, 0.0, 0.0, 1.0])
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (600.0, 601.0, 321.0, 239.0)
    raw = np.array([[1000, 0]], dtype=np.uint16)
    np.testing.assert_allclose(depth_to_meters(raw, '16UC1'), [[1.0, 0.0]])
    with pytest.raises(ValueError):
        depth_to_meters(raw, 'rgb8')


def test_boxes_and_masks():
    assert scale_box((0, 0, 10, 20), 0.5) == (2.5, 5.0, 7.5, 15.0)
    assert clip_box((-5.2, 3.7, 700.1, 479.5), 640, 480) == (0, 3, 640, 480)
    poly = np.array([[10, 10], [30, 10], [30, 20], [10, 20]], dtype=np.float32)
    mask = polygon_to_mask(poly, (50, 50))
    assert mask_centroid(mask) == pytest.approx((20.5, 15.5))
    assert polygon_to_mask(np.zeros((2, 2)), (50, 50)) is None


def test_depth_to_points_matches_deproject():
    depth = scene()
    depth[0, 0] = 0.0
    pts = depth_to_points(depth, INTR)
    assert np.isnan(pts[0, 0]).all()
    np.testing.assert_allclose(pts[150, 250], deproject(250, 150, 1.0, INTR), atol=1e-6)


def test_project_points_keeps_closest():
    pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [10.0, 0.0, 1.0]])
    img = project_points(pts, INTR, 480, 640)
    np.testing.assert_allclose(img[240, 320], [0.0, 0.0, 1.0])
    assert np.isfinite(img).all(axis=2).sum() == 1   # behind camera / outside dropped


def test_region_points_mask_and_roi():
    pts = depth_to_points(scene(obj=1.0, box=(225, 125, 275, 175)), INTR)
    box = (200, 100, 300, 200)
    roi = region_points(pts, box, roi_scale=0.5)
    assert roi.shape[0] == 50 * 50 and np.allclose(roi[:, 2], 1.0)
    mask = np.zeros((480, 640), bool)
    mask[130:140, 230:240] = True
    assert region_points(pts, box, mask=mask).shape[0] == 100


def test_region_mask_and_occluders():
    depth = scene(background=1.0, obj=0.9, box=(240, 180, 280, 220))   # cup on a book
    pts = depth_to_points(depth, INTR)
    book = region_mask(depth.shape, (200, 100, 400, 300), np.ones(depth.shape, bool))
    cup = region_mask(depth.shape, (240, 180, 280, 220), roi_scale=1.0)
    assert book.sum() == 200 * 200 and cup.sum() == 40 * 40
    book2, cup2 = exclude_occluders(pts, [book, cup], min_gap=0.03)
    assert book2.sum() == 200 * 200 - 40 * 40 and (cup2 == cup).all()
    assert (exclude_occluders(pts, [book, cup], min_gap=0.2)[0] == book).all()


def test_filter_object_points_drops_background():
    obj = np.column_stack([np.random.uniform(-0.05, 0.05, 300),
                           np.random.uniform(-0.05, 0.05, 300), np.full(300, 1.0)])
    bg = np.column_stack([np.random.uniform(-0.1, 0.1, 100),
                          np.random.uniform(-0.1, 0.1, 100), np.full(100, 3.0)])
    kept, stats = filter_object_points(np.vstack([obj, bg]), min_depth=0.1)
    assert stats.median == pytest.approx(1.0)
    assert kept.shape[0] == 300
    assert surface_centroid(kept)[2] == pytest.approx(1.0)
    assert filter_object_points(np.zeros((5, 3)), min_depth=0.1)[1] is None


def test_transform_points():
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))   # 90 deg about z
    np.testing.assert_allclose(quaternion_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    out = transform_points(np.array([[1.0, 0.0, 0.0]]), (0.0, 0.0, 1.0), q)
    np.testing.assert_allclose(out, [[0.0, 1.0, 1.0]], atol=1e-6)


def box_points(center, size, yaw, n=2000, seed=0):
    """Return points uniformly filling a yaw-rotated box."""
    rng = np.random.default_rng(seed)
    local = (rng.random((n, 3)) - 0.5) * np.asarray(size)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.asarray(center)


def test_fit_box_oriented():
    pts = box_points((1.0, 0.5, 0.8), (0.3, 0.1, 0.2), yaw=0.4)
    box = fit_box(pts, mode='yaw', low=0.0, high=100.0)
    assert box.center == pytest.approx((1.0, 0.5, 0.8), abs=0.01)
    assert box.size == pytest.approx((0.3, 0.1, 0.2), abs=0.01)
    assert box.yaw == pytest.approx(0.4, abs=0.02)
    aligned = fit_box(pts, mode='axis_aligned', low=0.0, high=100.0)
    assert aligned.yaw == 0.0 and aligned.size[0] > 0.3


def test_fit_box_yaw_range_and_long_side():
    box = fit_box(box_points((0, 0, 0), (0.1, 0.4, 0.1), yaw=0.0), low=0.0, high=100.0)
    assert box.size[0] > box.size[1]
    assert box.yaw == pytest.approx(math.pi / 2, abs=0.02) or \
        box.yaw == pytest.approx(-math.pi / 2, abs=0.02)
    assert -math.pi / 2 < box.yaw <= math.pi / 2


def det(view, cls, center, score=0.8, size=(0.1, 0.1, 0.1)):
    """Build a ViewDetection with 3D data."""
    d = ViewDetection(view=view, index=0, class_name=cls, score=score, box=(0, 0, 10, 10),
                      centroid_2d=(5, 5), points=box_points(center, size, 0.0, n=200))
    return complete_3d(d, 'yaw', 10)


def test_merge_views():
    dets = [det(0, 'cup', (1.0, 0.0, 0.8), score=0.9),
            det(1, 'cup', (1.02, 0.01, 0.8), score=0.7),    # same cup from view 1
            det(1, 'cup', (1.5, 0.0, 0.8)),                 # another cup
            det(0, 'bowl', (1.0, 0.0, 0.8))]                # other class, same place
    merged = merge_views(dets, merge_distance=0.1, size_factor=0.5, box_mode='yaw')
    cups = sorted([m for m in merged if m.class_name == 'cup'], key=lambda m: m.box3d.center[0])
    assert len(cups) == 2 and len(merged) == 3
    assert cups[0].views == [0, 1] and cups[0].score == 0.9 and cups[0].points.shape[0] == 400
    assert cups[1].views == [1]


def test_complete_3d_requires_points():
    d = ViewDetection(view=0, index=0, class_name='cup', score=0.5, box=(0, 0, 1, 1),
                      centroid_2d=(0, 0), points=np.zeros((3, 3)))
    assert not complete_3d(d, 'yaw', min_points=10).has_3d


def make_cloud(points, colors=None, height=1):
    """Build a PointCloud2 with xyz and optional packed rgb."""
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate('xyz')]
    rows = [list(p) for p in points]
    if colors is not None:
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
        rows = [list(p) + [int(r) << 16 | int(g) << 8 | int(b)]
                for p, (r, g, b) in zip(points, colors)]
    cloud = create_cloud(Header(frame_id='cam'), fields, rows)
    if height > 1:
        cloud.height, cloud.width = height, len(rows) // height
        cloud.row_step = cloud.width * cloud.point_step
    return cloud


def test_cloud_to_arrays_organized_with_color():
    pts = [(0.1 * i, 0.0, 1.0) for i in range(6)]
    pts[2] = (float('nan'),) * 3
    cols = [(255, 0, 0)] * 6
    points, bgr = cloud_to_arrays(make_cloud(pts, cols, height=2))
    assert points.shape == (2, 3, 3) and bgr.shape == (2, 3, 3)
    assert np.isnan(points[0, 2]).all()
    assert points[1, 0, 0] == pytest.approx(0.3)
    assert tuple(bgr[0, 0]) == (0, 0, 255)


def test_cloud_to_arrays_unorganized_without_color():
    points, bgr = cloud_to_arrays(make_cloud([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]))
    assert points.shape == (2, 3) and bgr is None
