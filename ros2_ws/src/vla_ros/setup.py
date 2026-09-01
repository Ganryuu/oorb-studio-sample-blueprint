from setuptools import find_packages, setup

package_name = "vla_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/policy.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OORB",
    maintainer_email="contact@oorb.io",
    description="ROS 2 nodes for vision-language-action policy inference.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy = vla_ros.policy_node:main",
            "demo_publisher = vla_ros.demo_publisher:main",
        ],
    },
)
