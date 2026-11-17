# yolo_rgbd_detector

ROS 2 (Jazzy) **object detection service** based on
[Ultralytics YOLO](https://docs.ultralytics.com/). You send one or more views
of the scene (color + aligned depth, or point clouds) and get back, for every
object:

- the 2D detection (box, mask centroid, class, score) in each image;
- an oriented 3D box in the frame you ask for (`target_frame`);
- a point on the visible surface, the number of 3D points used;
- the dominant color (`blue`, `white`, ...) and the mean RGB.

The node does not subscribe to cameras: it works only on request, so the
caller decides which images are processed and keeps them consistent with
other services (for example a grasp detector called on the same images).
The `scene_db` package uses it this way.

```
                   object_detection_msgs/DetectObjects
  client  ───────────────────────────────────────────────▶  yolo_rgbd_detector
          color[] + depth[] + camera_info[]   (case a)          │ YOLO (seg)
          point_clouds[]                      (case b)          │ points per object
          color[] + camera_info[] + clouds[]  (case c)          │ occluders removed
          target_frame, classes, min_score, merge_views         │ depth band filter
          ◀───────────────────────────────────────────────      │ oriented box, color
          Detection2DArray[] (per view)                         │ multi-view merge
          Detection3DArray   (target frame)
          DetectedObject[]   (everything)
```

## Features

- **Service only** (`~/detect`), no continuous processing.
- **Three input modes** per view: color + aligned depth + camera_info; an
  organized RGB point cloud; color + camera_info + any point cloud (projected
  into the image).
- **Several views in one call**, each with its own TF; optional **merging**
  of the same object seen from different views.
- **3D from points**, not from a single depth value: mask pixels → 3D points
  → occluders removed → depth band around the median → oriented box (yaw
  about the target z axis) or axis-aligned box.
- **Color** by per-pixel HSV vote on the mask.
- Debug image and RViz markers published after every call.

## Requirements

- ROS 2 Jazzy, Ubuntu 24.04, Python 3.12
- `vision_msgs`, `cv_bridge`, `tf2_ros`, `sensor_msgs_py`,
  `object_detection_msgs` (in this workspace)
- `ultralytics` (pip), `numpy < 2`

## Installation

```bash
# ROS dependencies
sudo apt install ros-jazzy-vision-msgs ros-jazzy-cv-bridge ros-jazzy-tf2-ros \
    ros-jazzy-sensor-msgs-py

# Ultralytics in a virtual environment that can still see the ROS Python packages.
# Keep numpy < 2: cv_bridge on Ubuntu 24.04 is built against numpy 1.x.
python3 -m venv --system-site-packages ~/venvs/yolo
source ~/venvs/yolo/bin/activate
pip install ultralytics "numpy<2"
touch ~/venvs/yolo/COLCON_IGNORE   # if the venv is inside a workspace

# Build
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to yolo_rgbd_detector
source install/setup.bash
```

The node uses the `python3` found on `PATH` (see `setup.cfg`), so **activate the
venv before launching it**. If you prefer not to use a venv, run
`pip install --break-system-packages ultralytics "numpy<2"` instead.

## Quick start

```bash
# Terminal 1: the service
source ~/venvs/yolo/bin/activate
ros2 launch yolo_rgbd_detector yolo_rgbd_detector.launch.py

# Terminal 2: a camera (RealSense example) and a TF from the camera to "world"
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
ros2 run tf2_ros static_transform_publisher --z 1.0 --frame-id world \
    --child-frame-id camera_link

# Terminal 3: call the service on the current images
ros2 run yolo_rgbd_detector detect_once --camera camera --target-frame world
```

`detect_once` grabs one synchronized color/depth/camera_info triple and prints
the objects; it is a minimal client, see its source and
[docs/TUTORIAL.md](docs/TUTORIAL.md) for writing your own. RViz: add
`/yolo_rgbd_detector/debug_image` (Image) and `/yolo_rgbd_detector/markers`
(MarkerArray).

To test without a camera, use the Gazebo simulation in
[`yolo_rgbd_sim`](../yolo_rgbd_sim).

## Interface summary

| Name | Type | Description |
|---|---|---|
| `~/detect` | service `object_detection_msgs/DetectObjects` | Detect objects in the given views |
| `~/debug_image` | topic `sensor_msgs/Image` | Annotated image of each view, after each call |
| `~/markers` | topic `visualization_msgs/MarkerArray` | 3D boxes and labels in the target frame |

Main parameters (see [docs/API.md](docs/API.md) for all of them):

| Parameter | Default | Meaning |
|---|---|---|
| `model` | `yolo11n-seg.pt` | YOLO weights (name or path); `-seg` models give masks |
| `confidence_threshold` | 0.5 | Default minimum score |
| `classes` | all | Default class filter |
| `box_mode` | `yaw` | `yaw` (rotated about target z) or `axis_aligned` |
| `exclude_occluders` | true | Remove the pixels of objects in front |
| `merge_distance` | 0.1 m | Gate for merging views |

## How the 3D box is computed

1. **Pixels**: the segmentation mask limited to the box (without masks: the
   central `roi_scale` part of the box).
2. **Occluders**: pixels of other detections whose median depth is closer by
   more than `min_depth_band` are removed (a cup standing on a book is not
   part of the book).
3. **Points**: each pixel becomes a 3D point (depth + intrinsics, or the
   point cloud).
4. **Depth band**: keep the points whose depth is within
   `max(min_depth_band, lateral extent)` of the median: background seen
   through the mask edges is dropped.
5. **Box**: the points are transformed to `target_frame`; the box encloses
   the 2nd–98th percentile, rotated about z by the minimum-area rectangle of
   the footprint (`x` = longer side).

Only the visible surface is measured: the box is shorter than the object
along the viewing direction. Several views (with `merge_views`) reduce this.

## Limitations

- One mask per object: an object fully surrounding another (a ring of pixels)
  still includes it unless the inner one is detected too.
- Yaw-only boxes: tilted objects get a larger box.
- Colors are coarse names from HSV thresholds, sensitive to lighting.
- Inference runs one request at a time.

## Testing

```bash
colcon test --packages-select yolo_rgbd_detector --event-handlers console_direct+
```

Unit tests cover geometry, point cloud parsing, multi-view merge and colors;
they do not need YOLO.

## Documentation

- [docs/API.md](docs/API.md): service, topics, parameters, Python modules
- [docs/TUTORIAL.md](docs/TUTORIAL.md): step-by-step usage, clients in Python

## License

Apache License 2.0, see [LICENSE](LICENSE).
