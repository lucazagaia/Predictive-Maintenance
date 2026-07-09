import os
from glob import glob

from setuptools import setup

package_name = "adr_pdm"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Luca Zagaia",
    maintainer_email="zagaialuca@gmail.com",
    description="rclpy nodes wrapping the ADR predictive-maintenance pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot_driver = adr_pdm.robot_driver_node:main",
            "edge_node = adr_pdm.edge_node:main",
            "cloud_node = adr_pdm.cloud_node:main",
        ],
    },
)
