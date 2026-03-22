import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_bot_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OORB Studio',
    maintainer_email='blueprints@oorb.io',
    description='Launch files and configs for 4-DOF robot arm',
    license='MIT',
)
