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

"""Per-view detections and their fusion across views. No ROS dependency."""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from yolo_rgbd_detector.colors import ColorEstimate
from yolo_rgbd_detector.geometry import Box, Box3D, box_distance, fit_box, surface_centroid


@dataclass
class ViewDetection:
    """One YOLO detection in one view, with its 3D data in the target frame."""

    view: int
    index: int                         # detection index in its view
    class_name: str
    score: float
    box: Box                           # 2D box (px)
    centroid_2d: Tuple[float, float]   # mask centroid or box center (px)
    color: Optional[ColorEstimate] = None
    points: Optional[np.ndarray] = None   # N x 3, target frame (None = no 3D)
    box3d: Optional[Box3D] = None
    surface: Optional[Tuple[float, float, float]] = None
    views: List[int] = field(default_factory=list)

    @property
    def id(self) -> str:  # noqa: A003
        """Id of the detection in the response: ``<view>_<index>``."""
        return f'{self.view}_{self.index}'

    @property
    def has_3d(self) -> bool:
        """Return True if the detection has a 3D box."""
        return self.box3d is not None


def complete_3d(det: ViewDetection, box_mode: str, min_points: int) -> ViewDetection:
    """Fit the 3D box and surface centroid from ``det.points`` (if enough points)."""
    if det.points is None or det.points.shape[0] < max(1, min_points):
        det.points, det.box3d, det.surface = None, None, None
        return det
    det.box3d = fit_box(det.points, mode=box_mode)
    det.surface = surface_centroid(det.points)
    return det


def merge_views(detections: List[ViewDetection], merge_distance: float,
                size_factor: float, box_mode: str) -> List[ViewDetection]:
    """
    Merge the 3D detections of the same object seen in different views.

    Two detections of the same class from different views are merged when
    their box centers are closer than
    ``max(merge_distance, size_factor * larger horizontal box side)``.
    Merging is greedy, highest score first: the merged object keeps the 2D
    data, id and color of its best detection, the union of the points (box
    refitted), and the list of views. Detections without 3D are kept as they
    are.
    """
    with_3d = sorted([d for d in detections if d.has_3d], key=lambda d: -d.score)
    without = [d for d in detections if not d.has_3d]
    clusters: List[List[ViewDetection]] = []
    for det in with_3d:
        for cluster in clusters:
            best = cluster[0]
            if best.class_name != det.class_name:
                continue
            if any(d.view == det.view for d in cluster):
                continue  # one detection per view in a cluster
            gate = max(merge_distance,
                       size_factor * max(best.box3d.size[0], det.box3d.size[0]))
            if box_distance(best.box3d, det.box3d) <= gate:
                cluster.append(det)
                break
        else:
            clusters.append([det])

    merged = []
    for cluster in clusters:
        best = cluster[0]
        if len(cluster) == 1:
            merged.append(replace(best, views=[best.view]))
            continue
        points = np.concatenate([d.points for d in cluster])
        out = replace(best, points=points, views=[d.view for d in cluster])
        out.box3d = fit_box(points, mode=box_mode)
        out.surface = surface_centroid(points)
        merged.append(out)
    return merged + [replace(d, views=[d.view]) for d in without]
