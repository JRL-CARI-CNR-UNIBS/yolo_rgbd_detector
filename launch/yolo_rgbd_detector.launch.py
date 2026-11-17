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
Launch the YOLO object detection service (~/detect).

    ros2 launch yolo_rgbd_detector yolo_rgbd_detector.launch.py model:=/path/best.pt
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('yolo_rgbd_detector'), 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML file with the node parameters'),
        DeclareLaunchArgument('model', default_value='',
                              description='Override the model parameter (empty = params_file)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use /clock (simulation)'),
        OpaqueFunction(function=_make_node),
    ])


def _make_node(context):
    parameters = [LaunchConfiguration('params_file').perform(context),
                  {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context) == 'true'}]
    model = LaunchConfiguration('model').perform(context)
    if model:
        parameters.append({'model': model})
    return [Node(
        package='yolo_rgbd_detector', executable='detector_node', name='yolo_rgbd_detector',
        output='screen', emulate_tty=True, parameters=parameters)]
