# Simulation Models Directory

This directory contains robot models and simulation scenes for MuJoCo.

## Directory Structure

- models/mjcf/ - MuJoCo MJCF/XML model files
- models/xml/ - Additional XML scene files
- urdf/ - URDF robot descriptions (convert to MJCF with mujoco.urdf2mjcf)
- generated/ - Auto-generated models from CAD/text2CAD
- scenes/ - Complete simulation scenes with multiple objects

## Built-in Models

- pendulum.xml - Simple inverted pendulum with motor
- robot_arm.xml - 4-DOF robot arm (matches frontend default)

## Loading Models

### In Terminal (Python):
```bash
python3 << 'PY'
import mujoco
model = mujoco.MjModel.from_xml_path("/workspace/sim/models/mjcf/pendulum.xml")
data = mujoco.MjData(model)
print(f"Model loaded: {model.nq} DOFs, {model.nu} actuators")
PY
```

### In Frontend:
Use the "Upload MJCF" button in the MuJoCo simulator panel to upload models from your local machine. They will be saved to this directory and available for selection.

## Converting URDF to MJCF

If you have a URDF file in /workspace/sim/urdf/:
```bash
# Basic conversion using MuJoCo's built-in converter
python3 << 'PY'
import mujoco
model = mujoco.MjModel.from_xml_path("/workspace/sim/urdf/my_robot.urdf")
# The model can now be used or saved to MJCF format
PY
```

Note: URDF conversion requires mesh files to be in the correct paths. If you encounter errors, check that:
- Mesh files are relative to the URDF location
- Paths use forward slashes
- package:// URIs are resolved correctly
