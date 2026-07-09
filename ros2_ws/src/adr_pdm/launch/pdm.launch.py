"""Launch the single-robot pipeline: robot_driver -> edge_node -> cloud_node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scenario = LaunchConfiguration("scenario")
    robot_id = LaunchConfiguration("robot_id")

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="degraded",
                              description="healthy | degraded"),
        DeclareLaunchArgument("robot_id", default_value="adr-001"),

        Node(package="adr_pdm", executable="robot_driver", name="robot_driver",
             output="screen",
             parameters=[{"robot_id": robot_id, "scenario": scenario, "period": 3.0}]),
        Node(package="adr_pdm", executable="edge_node", name="edge_node", output="screen"),
        Node(package="adr_pdm", executable="cloud_node", name="cloud_node", output="screen"),
    ])
