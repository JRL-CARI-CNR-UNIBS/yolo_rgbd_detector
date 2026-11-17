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
Grab one RGB-D frame from a camera and call the detection service.

Examples::

    ros2 run yolo_rgbd_detector detect_once --camera camera --target-frame world
    ros2 run yolo_rgbd_detector detect_once --classes cup bowl --min-score 0.6

Topics (RealSense layout, ``<camera>`` = ``--camera``):
``/<camera>/color/image_raw``, ``/<camera>/aligned_depth_to_color/image_raw``,
``/<camera>/color/camera_info``.
"""

import argparse
import sys
import time

import message_filters
from object_detection_msgs.srv import DetectObjects
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def build_parser() -> argparse.ArgumentParser:
    """Command line arguments."""
    p = argparse.ArgumentParser(prog='ros2 run yolo_rgbd_detector detect_once',
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='camera', help='camera namespace')
    p.add_argument('--service', default='/yolo_rgbd_detector/detect')
    p.add_argument('--target-frame', default='', help='frame of the 3D boxes')
    p.add_argument('--classes', nargs='*', default=[])
    p.add_argument('--min-score', type=float, default=0.0)
    p.add_argument('--timeout', type=float, default=10.0)
    return p


def grab_frame(node: Node, camera: str, timeout: float):
    """Wait for one synchronized color / depth / camera_info triple."""
    frame = []
    subs = [message_filters.Subscriber(node, Image, f'/{camera}/color/image_raw',
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(node, Image,
                                       f'/{camera}/aligned_depth_to_color/image_raw',
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(node, CameraInfo, f'/{camera}/color/camera_info',
                                       qos_profile=qos_profile_sensor_data)]
    sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.05)
    sync.registerCallback(lambda *msgs: frame or frame.extend(msgs))
    deadline = time.monotonic() + timeout
    while not frame and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    for s in subs:
        node.destroy_subscription(s.sub)
    return frame or None


def main(argv=None) -> None:
    """Entry point."""
    args = build_parser().parse_args(rclpy.utilities.remove_ros_args(argv or sys.argv)[1:])
    rclpy.init(args=argv)
    node = Node('detect_once')
    code = 1
    try:
        frame = grab_frame(node, args.camera, args.timeout)
        if frame is None:
            print(f'No synchronized images from /{args.camera} within {args.timeout} s')
            return
        color, depth, info = frame
        client = node.create_client(DetectObjects, args.service)
        if not client.wait_for_service(timeout_sec=args.timeout):
            print(f'Service {args.service} not available')
            return
        future = client.call_async(DetectObjects.Request(
            color_images=[color], depth_images=[depth], camera_infos=[info],
            target_frame=args.target_frame, classes=args.classes,
            min_score=args.min_score))
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout * 3)
        res = future.result()
        if res is None:
            print('No response')
            return
        print(('OK: ' if res.success else 'ERROR: ') + res.message)
        for o in res.objects:
            line = f'  {o.id:6s} {o.color:7s} {o.class_name:15s} {o.score:.2f}'
            if o.has_3d:
                c, s = o.detection_3d.bbox.center.position, o.detection_3d.bbox.size
                line += (f'  center ({c.x:+.3f}, {c.y:+.3f}, {c.z:+.3f})'
                         f'  size ({s.x:.3f}, {s.y:.3f}, {s.z:.3f})  [{o.num_points} pts]')
            else:
                line += '  (no 3D)'
            print(line)
        code = 0 if res.success else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(code)


if __name__ == '__main__':
    main()
