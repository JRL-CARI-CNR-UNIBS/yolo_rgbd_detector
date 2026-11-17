# yolo_rgbd_detector: tutorial

This tutorial goes from the first call of the detection service to a Python
client that combines several views. It assumes the package is built and the
Ultralytics venv exists (see the [README](../README.md#installation)).

No camera? Start the Gazebo simulation of
[`yolo_rgbd_sim`](../../yolo_rgbd_sim/docs/TUTORIAL.md): it provides the same
topics as a RealSense and already launches the detector.

## 1. Start the service

```bash
source ~/venvs/yolo/bin/activate
source ~/ros2_ws/install/setup.bash
ros2 launch yolo_rgbd_detector yolo_rgbd_detector.launch.py
```

The first start downloads `yolo11n-seg.pt`. When you see
`Ready: service ~/detect`, check it:

```bash
ros2 service list | grep detect
# /yolo_rgbd_detector/detect
ros2 interface show object_detection_msgs/srv/DetectObjects
```

Use another model with `model:=/path/to/best.pt`, or copy
`config/params.yaml`, edit it and pass `params_file:=...`.

## 2. One detection from a camera

Start a RealSense with the depth aligned to color:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

and call the service on the current frame with the bundled client:

```bash
ros2 run yolo_rgbd_detector detect_once --camera camera
```

```
OK: 2 objects (2 with 3D) in 1 view(s), frame 'camera_color_optical_frame'
  0_0    blue    cup             0.91  center (+0.052, +0.031, +0.648)  size (0.088, 0.081, 0.071)  [3120 pts]
  0_1    white   bowl            0.84  center (-0.120, +0.044, +0.701)  size (0.163, 0.152, 0.040)  [6020 pts]
```

Without `--target-frame` the boxes are in the camera optical frame
(z forward, y down), so "yaw" rotates about the viewing direction. For
manipulation use a frame with z up:

```bash
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 1.0 --pitch 1.5708 \
    --frame-id world --child-frame-id camera_link      # camera looking down
ros2 run yolo_rgbd_detector detect_once --camera camera --target-frame world
```

Other options: `--classes cup bowl`, `--min-score 0.7`.

In RViz (fixed frame `world`) add `/yolo_rgbd_detector/debug_image` and
`/yolo_rgbd_detector/markers`: they are refreshed after every call.

## 3. Reading the result

For each object you get:

- `detection_2d`: pixel box and 2D centroid (mask centroid);
- `detection_3d`: box in the target frame. The box covers the **visible
  surface**: seen from the side, a mug's box is shallower than the mug;
  seen from above, it is a thin slab at the height of the rim;
- `surface_centroid`: a robust point on the visible surface, good for
  pointing or approaching;
- `color`: dominant color name, `rgb`: mean color.

Objects without enough valid depth (`min_points`, default 20) have
`has_3d: false`: they are in `objects` and `detections_2d` but not in
`detections_3d`.

## 4. A Python client

```python
import rclpy
from rclpy.node import Node
from object_detection_msgs.srv import DetectObjects


class Client(Node):
    def __init__(self):
        super().__init__('my_detect_client')
        self.cli = self.create_client(DetectObjects, '/yolo_rgbd_detector/detect')
        self.cli.wait_for_service()

    def detect(self, color, depth, info, frame='world'):
        req = DetectObjects.Request(color_images=[color], depth_images=[depth],
                                    camera_infos=[info], target_frame=frame,
                                    classes=['cup', 'bottle'])
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if not res.success:
            raise RuntimeError(res.message)
        for o in res.objects:
            if o.has_3d:
                p = o.detection_3d.bbox.center.position
                print(f'{o.color} {o.class_name} at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        return res
```

`color`, `depth` and `info` come from your own subscriptions; synchronize
them with `message_filters.ApproximateTimeSynchronizer` (see
`yolo_rgbd_detector/detect_once.py`).

**Inside a node that also uses TF or other callbacks**, do not block in a
callback with `spin_until_future_complete`. Use a `MultiThreadedExecutor`,
put the client in a `ReentrantCallbackGroup` and wait on the future:

```python
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

self.cli = self.create_client(DetectObjects, '/yolo_rgbd_detector/detect',
                              callback_group=ReentrantCallbackGroup())
# in a callback:
future = self.cli.call_async(req)
while not future.done():
    time.sleep(0.005)
...
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
```

`scene_db/scene_updater.py` is a complete example: it calls the detector
and a grasp detector in parallel on the same images.

## 5. Point clouds

Organized RGB cloud only (case b), e.g. from `depth_image_proc` or a stereo
camera:

```python
req = DetectObjects.Request(point_clouds=[cloud], target_frame='world')
```

The image is rebuilt from the cloud's `rgb` field. An unorganized cloud
(e.g. a filtered or accumulated one, or a lidar in another frame) needs the
color image and its camera info (case c):

```python
req = DetectObjects.Request(color_images=[color], camera_infos=[info],
                            point_clouds=[cloud], target_frame='world')
```

The cloud is transformed into the camera frame with TF and projected into
the image; where several points fall in one pixel the closest is kept.

## 6. Several views

Pass one element per view in each array. Each view is transformed to
`target_frame` with TF at its own stamp, so the camera poses must be in TF
(a camera on a robot arm: take the images, the robot TF is recorded with
them).

```python
req = DetectObjects.Request(
    color_images=[c0, c1], depth_images=[d0, d1], camera_infos=[i0, i1],
    target_frame='world', merge_views=True)
```

With `merge_views` the same object seen in both views becomes one entry
with `views: [0, 1]` and a box fitted on the points of both views, which is
closer to the full object. Without it you get one entry per view and do the
fusion yourself.

## 7. Tuning

| Symptom | Try |
|---|---|
| Background in the box (box too deep) | Use a `-seg` model; lower `roi_scale` (without masks); lower `min_depth_band` |
| An object resting on another inflates its box | Keep `exclude_occluders: true`; both objects must be detected |
| Thin objects without 3D | Lower `min_points`; check depth holes |
| Boxes rotated about the wrong axis | Pass a `target_frame` with z up; or `box_mode: axis_aligned` |
| Wrong color names | Adjust the lighting; colors come from HSV thresholds (`colors.py`) |
| Slow first call | Normal: the model is loaded and warmed up on the first request |

## 8. Using it with the scene database

The [`scene_db`](../../scene_db) package keeps a persistent database of the
objects: its `scene_updater` node calls this service on request, associates
the results with the stored objects and writes them in one transaction:

```bash
ros2 launch scene_db scene_db.launch.py
ros2 run scene_db cli acquire --grasps
ros2 run scene_db cli list
```
