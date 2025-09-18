# Gino
## Leader free SO-100 teleop

A Python library to teleoperate SO-100(1) 5DoF robot arms without the need of a leader arm, using a smartphone as control device.

## How does it work

This is just if you have 5 minutes free and are curious about how it works. It's not needed if you just want to use the library.

### Background

The library is made up of two main parts:
- motion tracking
- robot control

While the latter was more straightforward, the pose estimation went through some iterations.

- The original idea was to use an IMU, getting rotation from the gyroscope and translation from the accelerometer.

   PROBLEM: I wasn´t able to retrieve translation values from the accelerometer, no matter what how I tryied.

   NOTE: tryied on 2 different sensors:
      - SENSORNAME from the Nintendo Switch (1) JoyCons
      - SENSORNAME from the Arduino Nano RP2040

- I then switched to ArUco markers which worked after implementing a simple sensor fusion algorithm to complement rotation values with the ones from a gyroscope VIDEO HERE.

   PROBLEM: in order to avoid motion blur (which capped the speed at which the controller cuold be used effectively) the ArUco cube needed to be backlit. I built a simple prototype PCB for that, but at that point the solution cuold not be easily replicated.

   NOTE: this version was likely the one that perfermed better from a smoothness/accuracy point of veiw.

- I finally settled on using AR APIs from Unity. Which is the current implementation. I implemented (read "ChatGPT implemented") a Unity AR app which sends absolute position and orientation values over Wi-Fi. The whole app is basically just made of an API call.

   NOTE: I didn´t even think such a thing was possible until I stumbled on LINK from LINK. Unfortunately their implementation relied on LINK, which is not supported on iOS. Since I wanted a cross-platform option*, after a bit of ChatGPTing, I settled on Unity.

* At the moment no iOS version is available. Unfortunately I have no iPhones and/or Macs and both are required for the developing of the former. If you have them at your disposal and would like to help build it, I would be more than happy to provide all the help you need (its by no mean a difficult process).

### Project Structure

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

### Project breakdown

Inside `main.py` the following actions are performed:

- `Receiver` class from `src/gino/udp/receiver.py` is used to get and unpack the data sent over Wi-Fi from the smartphone*.

- `MoveRobot` class from `src/gino/kinematics/move_robot.py` is used to perform the inverse kinematics calculations needed turn the 6D coordinates (pose + orientation) into arm joints states.

   - The class in turn relies on the `RobotKinematics` PINOCCHIOLINK wrapper from `src/gino/kinematics/kinematics.py`. Credits for the former code go to Joe Clinton with his LINK repository, which turned out to be really important for this project.

#### ArUco (deprecated)
As said previously an earlier iteration of the project relied of ArUco markers for the pose estimation. The related code is no longer used by the library itself but I thought that someone could find it useful for other project, hence left it here.

The related files are inside the LINK brach. Thery're not well documented so feel free to ask in case of need.

## Installation

### 0. LeRobot installation

You should already have created the `lerobot` conda env and installed the LeRobot library inside it.

Be sure to have the env activated:
```bash
conda activate lerobot
```

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Gino
```

### 3. Install Core Dependencies
```bash
conda install -c conda-forge numpy scipy matplotlib pyserial pinocchio
```

OPTIONAL: please note that if you wanted to experiment with the AruCo detection you'll need `opencv-contrib-python` as well. Install it as follows:
```bash
conda install -c conda-forge opencv
```

## Usage

### Basic Robot Control
The main application provides real-time robot control via UDP communication:

```bash
python main.py
```

## Configuration

### Robot Setup
- Ensure your robot is connected via serial port (default: `/dev/ttyACM0`)
- Configure the robot ID in the code (default: "toni")
- Verify URDF files are present in `data/urdf/`

### Network Configuration
- UDP receiver runs on port 5005 by default
- Ensure firewall allows UDP communication on this port

## Aknowledgements

As previously mentioned I would like to warmly thank:

- Joe Clinton for the LINK project. Which provided a strong base for the development of the inverse kinematics algorithm.
- SERBIANNAME for NAME which provided the idea behind the whole app.
- Inria for the Pinocchio library. Way too many times we think that research (expecially public one) is confined inside labs and never finds applications in the real world. This library and all the amazing things I discovered people create with it demonstrate that it is not the case.

Their work has been really important for the success of this project. I really hope that it too will be useful for someone and that people will build interesting things with it.

## License

This project is licensed under the terms specified in the LICENSE file.

# Link

This is Gino, a Python library I just realeased. It tryies to address one of the main issues with the SO-100(1) arms used extensively with the LeRobot library. These arms are a fantastic way to enter the what marketing departments like to call "embodied intelligence" which is none other than ML applied to robotics. They're cheap and easy to use and people are building amaizing applications out of them. The main drawback is the need for a leader arm, which acts as a controller for the follower arm, thus doubling the price. This library tryies to address it with an AR app and some Python code. If you want you can read more about it here: LINK.