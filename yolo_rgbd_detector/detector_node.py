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
ROS 2 object detection service: YOLO on one or more RGB-D views.

Service:
    ``~/detect``  object_detection_msgs/DetectObjects

Debug topics (published after every call):
    ``~/debug_image``  sensor_msgs/Image               annotated image of each view
    ``~/markers``      visualization_msgs/MarkerArray  3D boxes in the target frame

See ``docs/API.md`` for the full description.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
from object_detection_msgs.msg import DetectedObject
from object_detection_msgs.srv import DetectObjects
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from yolo_rgbd_detector.cloud import cloud_to_arrays
from yolo_rgbd_detector.colors import object_color
from yolo_rgbd_detector.fusion import complete_3d, merge_views, ViewDetection
from yolo_rgbd_detector.geometry import (
    depth_to_meters,
    depth_to_points,
    exclude_occluders,
    filter_object_points,
    Intrinsics,
    mask_centroid,
    polygon_to_mask,
    project_points,
    region_mask,
    region_points,
    transform_points,
)

# Ultralytics pulls in PyTorch, which is slow to import: import it lazily.
YOLO = None

IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def class_color(class_id: int) -> tuple:
    """Return a stable, distinct BGR color (0-255) for a class id."""
    hue = int((class_id * 0.618033988749895 % 1.0) * 179)
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


@dataclass
class View:
    """One input view, ready for detection."""

    index: int
    header: Header
    bgr: np.ndarray                        # H x W x 3 uint8
    points: Optional[np.ndarray] = None    # H x W x 3 points image (camera frame) or None


class RequestError(ValueError):
    """Invalid service request."""


class YoloRgbdDetector(Node):
    """Detect objects with YOLO and locate them in 3D."""

    def __init__(self) -> None:
        super().__init__('yolo_rgbd_detector')
        self._declare_parameters()
        self._p = self._read_parameters()

        global YOLO
        if YOLO is None:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO
        self.get_logger().info(f"Loading YOLO model '{self._p['model']}'")
        self._model = YOLO(self._p['model'])
        self._class_names: Dict[int, str] = dict(self._model.names)
        self._default_classes = self._resolve_classes(self._p['classes'])

        self._bridge = CvBridge()
        # The service waits for TF: with the MultiThreadedExecutor of main() the
        # TF subscriptions (reentrant group) keep running meanwhile.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub_debug = (self.create_publisher(Image, '~/debug_image', 5)
                           if self._p['publish_debug'] else None)
        self._pub_markers = (self.create_publisher(MarkerArray, '~/markers', 5)
                             if self._p['publish_debug'] else None)
        # One request at a time: inference is not thread-safe.
        self.create_service(DetectObjects, '~/detect', self._srv_detect,
                            callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f'Ready: service ~/detect, {len(self._class_names)} classes, '
            f"default filter: {self._p['classes'] or 'all'}")

    # ------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter('model', 'yolo11n-seg.pt')
        self.declare_parameter('device', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('classes', Parameter.Type.STRING_ARRAY)  # unset = all
        self.declare_parameter('use_mask', True)
        self.declare_parameter('roi_scale', 0.5)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('depth_scale', 0.0)
        self.declare_parameter('min_depth_band', 0.03)
        self.declare_parameter('min_points', 20)
        self.declare_parameter('exclude_occluders', True)
        self.declare_parameter('box_mode', 'yaw')
        self.declare_parameter('merge_distance', 0.1)
        self.declare_parameter('merge_size_factor', 0.5)
        self.declare_parameter('tf_timeout', 0.5)
        self.declare_parameter('publish_debug', True)

    def _read_parameters(self) -> dict:
        names = ['model', 'device', 'confidence_threshold', 'iou_threshold', 'image_size',
                 'use_mask', 'roi_scale', 'min_depth', 'max_depth', 'depth_scale',
                 'min_depth_band', 'min_points', 'exclude_occluders', 'box_mode', 'merge_distance',
                 'merge_size_factor', 'tf_timeout', 'publish_debug']
        p = {n: self.get_parameter(n).value for n in names}
        try:
            p['classes'] = [c for c in self.get_parameter('classes').value if c]
        except ParameterUninitializedException:
            p['classes'] = []
        if not 0.0 < p['roi_scale'] <= 1.0:
            raise ValueError(f"roi_scale must be in (0, 1], got {p['roi_scale']}")
        if p['box_mode'] not in ('yaw', 'axis_aligned'):
            raise ValueError(f"box_mode must be 'yaw' or 'axis_aligned', got {p['box_mode']}")
        return p

    def _resolve_classes(self, names: List[str]) -> Optional[List[int]]:
        """Map class names to model ids. None keeps all classes."""
        if not names:
            return None
        by_name = {v: k for k, v in self._class_names.items()}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise RequestError(f'Unknown classes {unknown}. Model classes: {sorted(by_name)}')
        return [by_name[n] for n in names]

    # ------------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------------
    def _srv_detect(self, req: DetectObjects.Request, res: DetectObjects.Response):
        try:
            views = self._prepare_views(req)
            classes = (self._resolve_classes(list(req.classes)) if req.classes
                       else self._default_classes)
            min_score = req.min_score if req.min_score > 0.0 else self._p['confidence_threshold']
            target = req.target_frame or views[0].header.frame_id

            per_view: List[List[ViewDetection]] = []
            annotated: List[np.ndarray] = []
            for view in views:
                tf = self._lookup(target, view.header) if view.points is not None else None
                dets = self._detect_view(view, classes, min_score, tf)
                per_view.append(dets)
                annotated.append(self._annotate(view, dets))
        except (RequestError, TransformException) as err:
            res.success, res.message = False, str(err)
            self.get_logger().warn(f'detect: {err}')
            return res

        flat = [d for dets in per_view for d in dets]
        objects = (merge_views(flat, self._p['merge_distance'], self._p['merge_size_factor'],
                               self._p['box_mode'])
                   if req.merge_views and len(views) > 1
                   else [replace(d, views=[d.view]) for d in flat])

        target_header = Header(stamp=views[0].header.stamp, frame_id=target)
        res.detections_2d = [
            Detection2DArray(header=v.header,
                             detections=[self._detection_2d(v.header, d) for d in dets])
            for v, dets in zip(views, per_view)]
        res.objects = [self._detected_object(views, target_header, d) for d in objects]
        res.detections_3d = Detection3DArray(
            header=target_header,
            detections=[o.detection_3d for o in res.objects if o.has_3d])
        n3d = len(res.detections_3d.detections)
        res.success = True
        res.message = (f'{len(res.objects)} objects ({n3d} with 3D) in {len(views)} view(s), '
                       f"frame '{target}'")
        self._publish_debug(views, annotated, target_header, objects)
        return res

    # ------------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------------
    def _prepare_views(self, req) -> List[View]:
        colors, depths = list(req.color_images), list(req.depth_images)
        infos, clouds = list(req.camera_infos), list(req.point_clouds)
        n = max(len(colors), len(clouds))
        if n == 0:
            raise RequestError('No input: give color_images and/or point_clouds')
        for name, arr in (('color_images', colors), ('depth_images', depths),
                          ('camera_infos', infos), ('point_clouds', clouds)):
            if arr and len(arr) != n:
                raise RequestError(f'{name} has {len(arr)} elements, expected 0 or {n}')

        views = []
        for i in range(n):
            color = colors[i] if colors else None
            depth = depths[i] if depths else None
            info = infos[i] if infos else None
            cloud = clouds[i] if clouds else None
            views.append(self._prepare_view(i, color, depth, info, cloud))
        return views

    def _prepare_view(self, i, color, depth, info, cloud) -> View:
        def intrinsics() -> Intrinsics:
            if info is None:
                raise RequestError(f'View {i}: camera_infos is required with color images')
            intr = Intrinsics.from_k(info.k)
            if not intr.is_valid():
                raise RequestError(f'View {i}: camera_info has zero focal length')
            return intr

        if color is not None:
            bgr = self._to_bgr(color)
            h, w = bgr.shape[:2]
            if depth is not None:                                   # case (a)
                raw = self._bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
                try:
                    depth_m = depth_to_meters(raw, depth.encoding, self._p['depth_scale'])
                except ValueError as err:
                    raise RequestError(f'View {i}: {err}')
                if depth_m.shape[:2] != (h, w):
                    raise RequestError(
                        f'View {i}: depth {depth_m.shape[:2]} and color {(h, w)} sizes '
                        'differ: the depth image must be aligned to the color image')
                return View(i, color.header, bgr, depth_to_points(depth_m, intrinsics()))
            if cloud is not None:                                   # case (c)
                intr = intrinsics()
                pts, _ = self._read_cloud(i, cloud)
                pts = pts.reshape(-1, 3)
                if cloud.header.frame_id and cloud.header.frame_id != color.header.frame_id:
                    t, q = self._lookup(color.header.frame_id, cloud.header)
                    pts = transform_points(pts, t, q)
                return View(i, color.header, bgr, project_points(pts, intr, h, w))
            return View(i, color.header, bgr, None)                # 2D only

        pts, bgr = self._read_cloud(i, cloud)                       # case (b)
        if pts.ndim != 3:
            raise RequestError(f'View {i}: an unorganized cloud needs a color image and '
                               'camera_info (case c)')
        if bgr is None:
            raise RequestError(f'View {i}: the organized cloud has no rgb/rgba field')
        return View(i, cloud.header, np.ascontiguousarray(bgr), pts)

    def _to_bgr(self, msg) -> np.ndarray:
        try:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as err:  # cv_bridge raises its own error types
            raise RequestError(f'Cannot convert color image ({msg.encoding}): {err}')

    @staticmethod
    def _read_cloud(i, cloud):
        try:
            return cloud_to_arrays(cloud)
        except ValueError as err:
            raise RequestError(f'View {i}: {err}')

    def _lookup(self, target: str, header: Header) -> Tuple[tuple, tuple]:
        """Return (translation, quaternion) from header.frame_id to target."""
        if not target or target == header.frame_id:
            return IDENTITY
        tf = self._tf_buffer.lookup_transform(
            target, header.frame_id, Time.from_msg(header.stamp),
            timeout=Duration(seconds=self._p['tf_timeout']))
        t, q = tf.transform.translation, tf.transform.rotation
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------------
    def _detect_view(self, view: View, classes, min_score, tf) -> List[ViewDetection]:
        result = self._model.predict(
            view.bgr, conf=min_score, iou=self._p['iou_threshold'],
            imgsz=self._p['image_size'], classes=classes,
            device=self._p['device'] or None, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        polygons = (result.masks.xy
                    if self._p['use_mask'] and result.masks is not None else None)

        p = self._p
        dets, masks = [], []
        for i in range(len(xyxy)):
            box = tuple(float(v) for v in xyxy[i])
            mask = polygon_to_mask(polygons[i], view.bgr.shape) if polygons is not None else None
            centroid = mask_centroid(mask) if mask is not None else None
            if centroid is None:
                centroid = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
            dets.append(ViewDetection(
                view=view.index, index=i,
                class_name=self._class_names.get(int(class_ids[i]), str(class_ids[i])),
                score=float(scores[i]), box=box, centroid_2d=centroid,
                color=object_color(view.bgr, box, mask=mask, roi_scale=p['roi_scale'])))
            masks.append(region_mask(view.bgr.shape, box, mask, p['roi_scale']))

        if view.points is None:
            return dets
        if p['exclude_occluders'] and len(masks) > 1:
            masks = exclude_occluders(view.points, masks, p['min_depth_band'])
        for det, mask in zip(dets, masks):
            pts = region_points(view.points, det.box, mask=mask)
            pts, _ = filter_object_points(pts, p['min_depth'], p['max_depth'],
                                          p['min_depth_band'])
            if pts.shape[0] > 0:
                det.points = transform_points(pts, *tf)
            complete_3d(det, p['box_mode'], p['min_points'])
        return dets

    # ------------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------------
    @staticmethod
    def _hypothesis(d: ViewDetection) -> ObjectHypothesisWithPose:
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = d.class_name
        hyp.hypothesis.score = d.score
        return hyp

    def _detection_2d(self, header: Header, d: ViewDetection) -> Detection2D:
        x1, y1, x2, y2 = d.box
        det = Detection2D(header=header, id=d.id)
        det.bbox.center.position.x = 0.5 * (x1 + x2)
        det.bbox.center.position.y = 0.5 * (y1 + y2)
        det.bbox.size_x, det.bbox.size_y = float(x2 - x1), float(y2 - y1)
        hyp = self._hypothesis(d)
        hyp.pose.pose.position.x, hyp.pose.pose.position.y = (float(c) for c in d.centroid_2d)
        hyp.pose.pose.orientation.w = 1.0
        det.results.append(hyp)
        return det

    def _detection_3d(self, header: Header, d: ViewDetection) -> Detection3D:
        det = Detection3D(header=header, id=d.id)
        b = d.box3d
        c, q = det.bbox.center, b.quaternion
        c.position.x, c.position.y, c.position.z = b.center
        c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w = q
        det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = b.size
        hyp = self._hypothesis(d)
        hyp.pose.pose = c
        det.results.append(hyp)
        return det

    def _detected_object(self, views, target_header, d: ViewDetection) -> DetectedObject:
        obj = DetectedObject(id=d.id, class_name=d.class_name, score=d.score,
                             views=[int(v) for v in d.views], has_3d=d.has_3d)
        obj.detection_2d = self._detection_2d(views[d.view].header, d)
        if d.has_3d:
            obj.detection_3d = self._detection_3d(target_header, d)
            obj.surface_centroid = Point(x=d.surface[0], y=d.surface[1], z=d.surface[2])
            obj.num_points = int(d.points.shape[0])
        if d.color is not None:
            obj.color = d.color.name
            r, g, b = d.color.rgb
            obj.rgb = ColorRGBA(r=r, g=g, b=b, a=1.0)
            obj.color_fraction = d.color.fraction
        return obj

    # ------------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------------
    def _annotate(self, view: View, dets: List[ViewDetection]) -> np.ndarray:
        if self._pub_debug is None:
            return view.bgr
        img = view.bgr.copy()
        by_name = {v: k for k, v in self._class_names.items()}
        for d in dets:
            bgr = class_color(by_name.get(d.class_name, 0))
            x1, y1, x2, y2 = (int(round(v)) for v in d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            u, v = (int(round(c)) for c in d.centroid_2d)
            cv2.drawMarker(img, (u, v), bgr, cv2.MARKER_CROSS, 14, 2)
            color = f'{d.color.name} ' if d.color else ''
            text = f'{d.id} {color}{d.class_name} {d.score:.2f}'
            text += '' if d.has_3d else ' (no 3D)'
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, th + base)
            cv2.rectangle(img, (x1, ty - th - base), (x1 + tw, ty), bgr, -1)
            cv2.putText(img, text, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def _publish_debug(self, views, annotated, header, objects) -> None:
        if self._pub_debug is None:
            return
        for view, img in zip(views, annotated):
            msg = self._bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header = view.header
            self._pub_debug.publish(msg)

        markers = MarkerArray()
        markers.markers.append(Marker(header=header, action=Marker.DELETEALL))
        for i, d in enumerate(o for o in objects if o.has_3d):
            b = d.box3d
            rgb = d.color.rgb if d.color else (0.6, 0.6, 0.6)
            cube = Marker(header=header, ns='boxes', id=i, type=Marker.CUBE,
                          action=Marker.ADD)
            cube.pose.position = Point(x=b.center[0], y=b.center[1], z=b.center[2])
            (cube.pose.orientation.x, cube.pose.orientation.y,
             cube.pose.orientation.z, cube.pose.orientation.w) = b.quaternion
            cube.scale.x, cube.scale.y, cube.scale.z = b.size
            cube.color = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=0.45)
            label = Marker(header=header, ns='labels', id=i, type=Marker.TEXT_VIEW_FACING,
                           action=Marker.ADD)
            label.pose.position = Point(x=b.center[0], y=b.center[1],
                                        z=b.center[2] + 0.5 * b.size[2] + 0.05)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.05
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f'{d.class_name} {d.score:.2f}'
            markers.markers.extend([cube, label])
        self._pub_markers.publish(markers)


def main(args=None) -> None:
    """Run the detector service."""
    rclpy.init(args=args)
    node = None
    try:
        node = YoloRgbdDetector()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
