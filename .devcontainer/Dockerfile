# Replicates OORB Studio's container environment for local development.
# If this Dockerfile works, the same project works in Studio. And vice versa.
FROM ros:jazzy

# System tools (matches Studio base image)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    # ROS packages this Blueprint needs
    ros-jazzy-joint-state-publisher \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-xacro \
    ros-jazzy-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --break-system-packages \
    mujoco>=3.0 \
    numpy

# Auto-source ROS on every shell (matches Studio .bashrc)
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "[ -f /workspace/ros2_ws/install/setup.bash ] && source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /workspace
