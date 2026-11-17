^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package yolo_rgbd_detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2026-11-17)
------------------
* First release.
* ``~/detect`` service (``object_detection_msgs/DetectObjects``) based on
  Ultralytics YOLO, no continuous camera subscription.
* Three input modes per view: color + aligned depth + camera_info, organized
  RGB point cloud, color + camera_info + any point cloud.
* Multiple views per call, each with its own TF, with optional merging of the
  same object seen from different views.
* Oriented 3D boxes in ``target_frame`` computed from mask points, with
  occluder removal and depth band filtering.
* Dominant color name and mean RGB for each object.
* ``detect_once`` command-line client, launch file and parameter file.
* Documentation: ``docs/API.md`` and ``docs/TUTORIAL.md``.
