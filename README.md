# Gino - Robot Kinematics, Vision, and Control

A comprehensive Python library for robot kinematics, computer vision, and real-time robot control. This project provides tools for ArUco marker detection, robot kinematics calculations, orientation fusion, and real-time robot manipulation.

## Features

- **Robot Kinematics**: Forward and inverse kinematics using Pinocchio
- **Computer Vision**: ArUco marker detection and tracking
- **Real-time Control**: UDP communication for robot control
- **Orientation Fusion**: Gyroscope and vision-based orientation estimation
- **Robot Visualization**: 3D robot model visualization and trajectory plotting
- **Camera Calibration**: Camera calibration utilities

## Dependencies

### Core Dependencies
The following dependencies are automatically installed with the package:

- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `opencv-contrib-python` - Computer vision (with ArUco support)

### Additional Dependencies

#### Pinocchio (Required for Kinematics)
Pinocchio is required for robot kinematics calculations but must be installed manually:

```bash
conda install -c conda-forge pinocchio
```

#### LeRobot (Required for Robot Control)
LeRobot is required for robot communication and control:

```bash
pip install lerobot
```

#### Additional System Dependencies
- **Serial Communication**: For robot communication via `/dev/ttyACM0`
- **UDP Network**: For real-time control communication
- **Camera Access**: For computer vision applications

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Gino
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Core Dependencies
```bash
pip install -e .
```

### 4. Install Pinocchio
```bash
conda install -c conda-forge pinocchio
```

### 5. Install LeRobot
```bash
pip install lerobot
```

## Usage

### Basic Robot Control
The main application provides real-time robot control via UDP communication:

```bash
python main.py
```

### Running Tests
The project includes comprehensive tests for different functionalities:

```bash
# Test kinematics
python tests/test_kinematics.py

# Test ArUco detection
python tests/test_10_aruco_orientation.py

# Test robot movement
python tests/test_6_move_robot_joints.py

# Test real-time control
python tests/test_8_real_move_real_time.py
```

### Camera Calibration
To calibrate your camera for ArUco detection:

```bash
# Collect calibration images
python calibration/collect_imgs.py

# Perform camera calibration
python calibration/calibrate_camera.py
```

### Examples
Check the `examples/` directory for usage examples:

```bash
# Orientation fusion example
python examples/orientation_fusion_example.py

# Simple orientation fusion
python examples/simple_orientation_fusion_example.py
```

## Configuration

### Robot Setup
- Ensure your robot is connected via serial port (default: `/dev/ttyACM0`)
- Configure the robot ID in the code (default: "toni")
- Verify URDF files are present in `data/urdf/`

### Network Configuration
- UDP receiver runs on port 5005 by default
- Ensure firewall allows UDP communication on this port

### Camera Setup
- Place ArUco markers in the robot workspace
- Ensure good lighting for marker detection
- Run camera calibration for accurate pose estimation

## Project Structure

```
Gino/
├── src/gino/           # Main source code
│   ├── aruco/         # ArUco marker detection
│   ├── kinematics/    # Robot kinematics
│   ├── mcu/          # Microcontroller communication
│   └── udp/          # UDP communication
├── tests/            # Test files
├── examples/         # Usage examples
├── calibration/      # Camera calibration tools
├── data/            # Data files and URDF models
└── resources/       # Documentation and resources
```

## Troubleshooting

### Common Issues

1. **Pinocchio Import Error**
   - Ensure Pinocchio is installed via conda: `conda install -c conda-forge pinocchio`

2. **Robot Connection Issues**
   - Check serial port permissions: `sudo chmod 666 /dev/ttyACM0`
   - Verify robot is powered and connected

3. **Camera Issues**
   - Ensure camera permissions are granted
   - Check if camera is being used by another application

4. **UDP Communication Issues**
   - Verify firewall settings
   - Check if port 5005 is available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or support, contact Giovanni Pegoraro at giovipegoraro@gmail.com.

