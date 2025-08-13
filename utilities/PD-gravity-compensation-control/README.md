# PD-Gravity-Compensation-Control

A Python application for controlling the joints of your robot (real or simulated) using a PD (Proportional-Derivative) controller with gravity compensation.

## Features

- **PD Control with Gravity Compensation**: Control for selected joints with dynamic gravity compensation
- **Safety Monitoring**: Real-time checks for joint limits and motor currents
- **Interactive Parameter Tuning**: Runtime parameter adjustment via YARP RPC interface
- **Data Logging**: Comprehensive logging of control references and gains
- **Multi-Platform Support**: Compatible with both real robots and Gazebo simulation
- **Configurable Joint Selection**: Control only the joints you need

## Repository Structure

```
PD-gravity-compensation-control/
├── blf-PD-gravity-compensation-control.py    # Main application script
├── robots/                                   # Robot configuration files
│   ├── ergoCubSN001/                        # Real robot configuration
│   └── ergoCubGazeboV1_1/                   # Gazebo simulation configuration
├── CMakeLists.txt                           # CMake install rules
└── README.md                                # You are here
```

## Installation

If bipedal locmotion framework is installed correctly, the application can be run from anywhere. In order to install the application, you should set the option FRAMEWORK_COMPILE_PDGravityCompensationControlApplication to ON.

## Configuration

Configuration files allow to specify the motors to drive and the current trajectory to use. [blf-PD-gravity-compensation-control.ini](https://github.com/fils99/bipedal-locomotion-framework/blob/2632c938486d63900d98ea712ca8356f345549ab/utilities/PD-gravity-compensation-control/config/robots/ergoCubGazeboV1_1/blf-PD-gravity-compensation-control-options.ini) is an example of the main configuration file.

In this file, the following parameters are defined:

- `dt`: the sample time of the controller
- `safety_threshold` the joints position safety threshold safety_threshold, which restricts the motion range of joints to [joint_lower_limit + safety_threshold, joint_upper_limit - safety_threshold]
- `compensation_bias_forces_factor`: the factor that multiplies the gravity torque contribution in the control law. It is probably about to be deprecated and set to 1.0 by default
- `joint_position_desired`: the desired position for the controlled joints
- `joint_velocity_desired`: the desired velocity for the controlled joints
- `Kp`: proportional gain of the controller
- `Kd`: derivative gain of the controller
- the `software_joint_lower_limits` optional float vector, which overrides (only if reducing the range of motion) the hardware joint_lower_limits.
- the `software_joint_upper_limits` optional float vector, which overrides (only if reducing the range of motion) the hardware joint_upper_limits.

- the `MOTOR` parameters group, which defines the list of available joints joints_list, the respective k_tau coefficient (in [A/Nm]) and the respective max_safety_current (in [A]).

Instead, the configuration file [robot_control.ini](https://github.com/fils99/bipedal-locomotion-framework/blob/2632c938486d63900d98ea712ca8356f345549ab/utilities/PD-gravity-compensation-control/config/robots/ergoCubGazeboV1_1/blf_PD_gravity_compensation_control/robot_control.ini) defines the list of motors to command.

If you want to run the application for a different robot remember to create a new folder in ./config/robots/. The name of the folder should match the name of the robot.


## Usage

### Basic Operation

It is a command line application, which can be run like:
```bash
blf-PD-gravity-compensation-control.py.
```

### Runtime Control via YARP RPC

The controller provides an interactive RPC interface for real-time parameter adjustment:

```bash
yarp rpc /PD_gravity_compensation_control/commands
```

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `setKp <joint> <value>` | Set proportional gain for a joint | `setKp l_hip_roll 50.0` |
| `setKd <joint> <value>` | Set derivative gain for a joint | `setKd l_hip_roll 2.0` |
| `getKp <joint>` | Get current proportional gain | `getKp l_hip_roll` |
| `getKd <joint>` | Get current derivative gain | `getKd l_hip_roll` |
| `setPos <joint> <angle>` | Set desired position (degrees) | `setPos l_hip_roll 30.0` |
| `getPos <joint>` | Get current desired position | `getPos l_hip_roll` |
| `show` | Display all current parameters | `show` |
| `help` | Show available commands | `help` |


### Data Logging

Moreover, the application streams data, which are the desired joint position and velocity, the gains $Kp$ and $Kd$, and the control mode, through the port /PD_gravity_compensation_control/logger/data:o. The user may collect the data via [YarpRobotLoggerDevice](https://github.com/fils99/bipedal-locomotion-framework/tree/add_new_PINN_architecture/devices/YarpRobotLoggerDevice).
