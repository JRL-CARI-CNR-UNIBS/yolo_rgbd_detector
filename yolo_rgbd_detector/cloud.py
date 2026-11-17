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

"""Read sensor_msgs/PointCloud2 into numpy arrays (xyz and color)."""

from typing import Optional, Tuple

import numpy as np
from sensor_msgs_py.point_cloud2 import dtype_from_fields


def cloud_to_arrays(msg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert a PointCloud2 to numpy arrays.

    :param msg: sensor_msgs/PointCloud2 with x, y, z (float32) fields and,
        optionally, a packed ``rgb`` or ``rgba`` field (PCL convention).
    :return: ``(points, bgr)``. For an organized cloud (height > 1) the
        shapes are H x W x 3 (a points image) and H x W x 3 uint8; otherwise
        N x 3 and N x 3. ``bgr`` is None without a color field. Invalid
        points are NaN.
    :raises ValueError: If x, y or z is missing or the cloud is big-endian.
    """
    names = [f.name for f in msg.fields]
    if not all(n in names for n in ('x', 'y', 'z')):
        raise ValueError(f'Point cloud without x/y/z fields (fields: {names})')
    if msg.is_bigendian:
        raise ValueError('Big-endian point clouds are not supported')

    dtype = dtype_from_fields(msg.fields, point_step=msg.point_step)
    n = msg.width * msg.height
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.row_step)
    data = np.ascontiguousarray(rows[:, :msg.width * msg.point_step]).view(dtype).reshape(n)

    points = np.stack([data['x'], data['y'], data['z']], axis=-1).astype(np.float32)
    points[~np.isfinite(points).all(axis=1)] = np.nan

    bgr = None
    color_field = 'rgb' if 'rgb' in names else ('rgba' if 'rgba' in names else None)
    if color_field is not None:
        packed = np.ascontiguousarray(data[color_field]).view(np.uint32)
        bgr = np.stack([packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF],
                       axis=-1).astype(np.uint8)

    if msg.height > 1:
        points = points.reshape(msg.height, msg.width, 3)
        if bgr is not None:
            bgr = bgr.reshape(msg.height, msg.width, 3)
    return points, bgr
