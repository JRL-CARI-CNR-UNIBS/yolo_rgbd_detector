# yolo_rgbd_detector: API reference

Node name: `yolo_rgbd_detector` (executable `detector_node`).

The node only works on request: it has no camera subscriptions.

## Service `~/detect` (`object_detection_msgs/DetectObjects`)

### Request

| Field | Type | Description |
|---|---|---|
| `color_images` | `sensor_msgs/Image[]` | Color image of each view (`bgr8`, `rgb8`, `mono8`, ... any `cv_bridge` encoding) |
| `depth_images` | `sensor_msgs/Image[]` | Depth **aligned to the color image** (same size): `16UC1` (mm) or `32FC1` (m) |
| `camera_infos` | `sensor_msgs/CameraInfo[]` | Intrinsics of the color camera (`k`) |
| `point_clouds` | `sensor_msgs/PointCloud2[]` | Point cloud of each view |
| `target_frame` | `string` | Frame of the 3D outputs. Empty = frame of view 0 |
| `classes` | `string[]` | Keep only these class names. Empty = `classes` parameter |
| `min_score` | `float32` | Minimum score. 0 = `confidence_threshold` parameter |
| `merge_views` | `bool` | Merge detections of the same object in different views |

Each array is either empty or has one element per view. A view `i` is one of:

| Case | Inputs | Notes |
|---|---|---|
| (a) | `color_images[i]`, `depth_images[i]`, `camera_infos[i]` | The usual RGB-D camera case. Depth must be registered to color (RealSense `align_depth`) |
| (b) | `point_clouds[i]` | Organized cloud (`height > 1`) with `x`, `y`, `z` and packed `rgb`/`rgba` (PCL convention); the image is rebuilt from the cloud |
| (c) | `color_images[i]`, `camera_infos[i]`, `point_clouds[i]` | Any cloud, organized or not, possibly in another frame (TF at the cloud stamp). Points are projected into the color image; the closest point per pixel wins |

A view with only a color image and camera info gives 2D detections and
colors, without 3D. When both depth and cloud are given, depth is used.

Each view is transformed to `target_frame` with TF at the stamp of its
header (`tf_timeout`), so views taken from different camera poses can be
combined if TF knows the poses.

### Response

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | false on invalid input or TF error (no partial results) |
| `message` | `string` | Summary or error description |
| `detections_2d` | `vision_msgs/Detection2DArray[]` | One array per view, header = view header |
| `detections_3d` | `vision_msgs/Detection3DArray` | Objects with 3D, `header.frame_id` = target frame, stamp of view 0 |
| `objects` | `object_detection_msgs/DetectedObject[]` | All objects with all the information |

`Detection2D`:

- `id`: `"<view>_<index>"`, same as the `DetectedObject.id`.
- `bbox`: axis-aligned box in pixels.
- `results[0].hypothesis`: `class_id` (class **name**, e.g. `"cup"`) and `score`.
- `results[0].pose.pose.position.{x, y}`: 2D centroid (mask centroid with
  segmentation models, otherwise box center).

`Detection3D`:

- `bbox.center`: box center and orientation (rotation about the z axis of the
  target frame only). `bbox.size`: `x` = longer horizontal side, `z` = height.
- `results[0]`: class, score, pose = box center.

`DetectedObject` adds:

| Field | Description |
|---|---|
| `views` | Views where the object was seen (several after merging; `views[0]` is the best detection) |
| `has_3d` | false if the object had fewer than `min_points` valid points |
| `surface_centroid` | Coordinate-wise median of the object's points (on the visible surface), target frame |
| `num_points` | Points used for the box |
| `color`, `rgb`, `color_fraction` | Dominant color name, mean RGB of those pixels in [0, 1], share of pixels |

### Multi-view merge

With `merge_views: true` and more than one view, 3D detections of the same
class from different views are merged when their box centers are closer
than `max(merge_distance, merge_size_factor * larger box side)`. Greedy,
highest score first, at most one detection per view in a merged object.
The merged object keeps id, 2D data and color of its best detection; its
box is refitted on the union of the points. Detections without 3D are never
merged.

### Errors

`success: false` with a message for: no input; arrays of different lengths;
missing camera info; zero focal length; unsupported depth encoding; depth and
color of different sizes; unorganized cloud without color image; organized
cloud without color field; unknown class names; TF lookup failure.

## Topics (published after every successful call)

| Topic | Type | Description |
|---|---|---|
| `~/debug_image` | `sensor_msgs/Image` | One annotated image per view (box, centroid, `id color class score`) |
| `~/markers` | `visualization_msgs/MarkerArray` | Namespace `boxes` (cubes colored with the measured RGB) and `labels`, target frame. Previous markers are deleted |

Disable both with `publish_debug: false`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `yolo11n-seg.pt` | Ultralytics weights: a name (downloaded on first use) or a path |
| `device` | string | `""` | `""` = auto, `cpu`, `cuda:0`, ... |
| `confidence_threshold` | double | 0.5 | Default `min_score` |
| `iou_threshold` | double | 0.45 | Non-maximum suppression |
| `image_size` | int | 640 | Inference size |
| `classes` | string[] | unset | Default class filter (unset = all classes) |
| `use_mask` | bool | true | Use segmentation masks when available |
| `roi_scale` | double | 0.5 | Without mask: fraction of the box used, (0, 1] |
| `min_depth` | double | 0.1 | Points closer than this (camera z, m) are discarded |
| `max_depth` | double | 10.0 | Points farther than this are discarded |
| `depth_scale` | double | 0.0 | Meters per raw depth unit; 0 = auto (`16UC1` → 0.001, `32FC1` → 1) |
| `min_depth_band` | double | 0.03 | Minimum half-width of the depth band (m); also the occluder gap |
| `min_points` | int | 20 | Minimum points for a 3D box |
| `exclude_occluders` | bool | true | Remove the pixels of detections that are in front |
| `box_mode` | string | `yaw` | `yaw` or `axis_aligned` (in the target frame) |
| `merge_distance` | double | 0.1 | Merge gate (m) |
| `merge_size_factor` | double | 0.5 | Merge gate factor on the box size |
| `tf_timeout` | double | 0.5 | Seconds to wait for each transform |
| `publish_debug` | bool | true | Publish `~/debug_image` and `~/markers` |

Parameters are read at startup.

## Executables

| Executable | Description |
|---|---|
| `detector_node` | The service node |
| `detect_once` | Command line client: grabs one frame from a camera namespace and prints the objects (`--camera`, `--target-frame`, `--classes`, `--min-score`, `--service`, `--timeout`) |

## Launch file

`yolo_rgbd_detector.launch.py` arguments: `params_file` (default
`config/params.yaml`), `model` (overrides the parameter), `use_sim_time`.

## Python modules

All modules except `detector_node` and `detect_once` are ROS-free and unit tested.

### `yolo_rgbd_detector.geometry`

| Function / class | Description |
|---|---|
| `Intrinsics(fx, fy, cx, cy)`, `Intrinsics.from_k(k)` | Pinhole intrinsics |
| `depth_to_meters(raw, encoding, scale=0)` | Raw depth image to meters (invalid = 0) |
| `deproject(u, v, z, intr)` | One pixel to a 3D point |
| `depth_to_points(depth_m, intr)` | H×W×3 points image, NaN where invalid |
| `project_points(points, intr, h, w)` | N×3 points to an H×W×3 points image (z-buffer) |
| `scale_box`, `clip_box`, `polygon_to_mask`, `mask_centroid` | 2D helpers |
| `region_mask(shape, box, mask, roi_scale)` | Pixels used for a detection |
| `exclude_occluders(points_img, masks, min_gap)` | Remove pixels of closer detections |
| `region_points(points_img, box, mask, roi_scale)` | Valid points of a detection |
| `filter_object_points(points, min_depth, max_depth, min_band)` | Depth band filter → `(points, DepthStats)` |
| `surface_centroid(points)` | Coordinate-wise median |
| `quaternion_matrix`, `transform_points`, `yaw_to_quaternion` | Transforms |
| `Box3D(center, yaw, size)`, `fit_box(points, mode, low, high, min_size)` | Oriented box fit |
| `box_distance(a, b)` | Distance between box centers |

### `yolo_rgbd_detector.cloud`

`cloud_to_arrays(msg) -> (points, bgr)`: PointCloud2 to numpy (H×W×3 for
organized clouds, N×3 otherwise); `bgr` is None without color field.

### `yolo_rgbd_detector.fusion`

`ViewDetection` (one detection in one view), `complete_3d(det, box_mode,
min_points)`, `merge_views(detections, merge_distance, size_factor, box_mode)`.

### `yolo_rgbd_detector.colors`

`COLOR_NAMES`, `ColorThresholds`, `classify_pixels(bgr)`,
`dominant_color(pixels_bgr)`, `object_color(image_bgr, box, mask, roi_scale,
max_pixels) -> ColorEstimate(name, rgb, fraction)`.

Classification per pixel (HSV, S and V in [0, 1], hue in degrees):

1. by hue: red < 15 or ≥ 345, orange 15–40, yellow 40–70, green 70–165,
   cyan 165–200, blue 200–260, purple 260–290, pink 290–345;
2. red or orange with V < 0.6 → `brown`;
3. S < 0.22 → `gray`, or `white` if also V ≥ 0.75;
4. V < 0.18 → `black` (overrides everything).

The most frequent class among the object pixels wins.
