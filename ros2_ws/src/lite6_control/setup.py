from setuptools import setup

package_name = 'lite6_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OORB Studio',
    maintainer_email='blueprints@oorb.io',
    description='UFactory Lite 6 control nodes',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'joint_publisher = lite6_control.joint_publisher:main',
            'mujoco_sim = lite6_control.mujoco_sim_node:main',
            'talker = lite6_control.talker:main',
        ],
    },
)
