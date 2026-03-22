from setuptools import find_packages, setup

package_name = "my_bot_py"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="developer",
    maintainer_email="developer@oorb.studio",
    description="OORB ROS2 demo package with robot arm joint publisher",
    license="MIT",
    entry_points={
        "console_scripts": [
            "talker = my_bot_py.talker:main",
            "robot_arm = my_bot_py.robot_arm_publisher:main",
            "mujoco_sim = my_bot_py.mujoco_sim_node:main",
        ],
    },
)
