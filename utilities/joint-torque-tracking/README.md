# joint-torque-tracking
The **joint-torque-tracking** is an application that allows to send torque commands
to the motors. The user can leverage predefined signals to define the reference torque.

## 🏃 How to use the application
It is a command line application, which can be run like:

```sh
blf-joint-torque-tracking.py
```

The application accepts an additional argument to specify the type of trajectory to deploy.
From terminal, run:

```sh
blf-joint-torque-tracking.py -h
```

to display the help message which details how to pass the additional input argument.

If bipedal locmotion framework is installed correctly, the application can be run from anywhere. In order to install the application, you should set the option `FRAMEWORK_COMPILE_JointTorqueTrackingApplication` to `ON`.

Once run, the application will:

1. Generate the torque trajectory used to drive the motors.

2. Drive the motors in torque as long as they are within the safety limits, which are defined in the configuration files.

In fact, the application will check if the joints come close to the position limits. In this case, the related motors are switch to position control
and kept still in the last read position as long as all the other motors have completed the torque trajectory.

Moreover, the application streams data, which is the joints reference torque and the control mode, through the port
`/joint_torque_tracking/logger/data:o`. The user may collect the data via [`YarpRobotLoggerDevice`](../../devices/YarpRobotLoggerDevice).

## Configuration files
Configuration files allow to specify the motors to drive and the torque trajectory to use.
[`blf-joint-torque-tracking-options.ini`](./config/robots/ergoCubSN001/blf-joint-torque-tracking-options.ini) is an example of the main configuration file.

In this file, the following parameters are defined:

1. the sample time `dt` of the reference trajectory
2. the joints position safety threshold `safety_threshold`, which restricts the motion range of joints to `[joint_lower_limit + safety_threshold, joint_upper_limit - safety_threshold]`
3. the `use_safety_threshold_for_starting_position` boolean, which determines wheter to use (or not) the `safety_threshold` when computing the starting positions.
4. the number of starting positions `number_of_starting_points` from which the same reference trajectory is repeated.
5. the `bypass_joint_torque_measure` boolean vector (one element per motor), which determines wether to use the measured joint torque when generating the trajectory as detailed later.
6. the `software_joint_lower_limits` optional float vector, which overrides (only if reducing the range of motion) the hardware `joint_lower_limits`.
7. the `software_joint_upper_limits` optional float vector, which overrides (only if reducing the range of motion) the hardware `joint_upper_limits`.
8. the `SINUSOID` parameters group, which defines the sinusoid trajectory as detailed later.
9. the `RAMP` parameters group, which defines the ramp trajectory as detailed later.
10. the `MOTOR` parameters group, which defines the list of available joints `joints_list`, the respective `k_tau` coefficient (in [A/Nm]) and the respective `max_safety_current` (in [A]).

Instead, the configuration file [`robot_control.ini`](./config/robots/ergoCubSN001/blf_joint_torque_tracking/robot_control.ini) defines the list of motors to command.

If you want to run the application for a different robot remember to create a new folder in
[`./config/robots/`](./config/robots). The name of the folder should match the name of the robot.

## Predefined Trajectories
Two trajectory types are currently supported: sinusoids and ramps.

### Sinusoids

They are defined as sinusoidal signals with varying frequency `f` and amplitude `A`:

```math

torque = torque_0 + A \sin (2 \pi f t)

```

The frequency is gradually decreased, every 2 cycles, from `max_frequency` to `min_frequency` by `frequency_decrement`.
Once the frequency range is covered the amplitude is increased by `delta_torque_increment`.
The starting amplitude is set by `min_delta_torque`, while the final one by `max_delta_torque`.

For each motor, if the respecitive boolean in the `bypass_joint_torque_measure` vector is set to `true`, then the initial torque `torque_0` is not the measured one at the starting position, but is set to `0`.

Each torque sinusoidal signal is repeated for each starting position.

These starting positions are computed by dividing the range of motion of the joint
(`[joint_lower_limit, joint_upper_limit]`) in `number_of_starting_points + 1` segments, and
then taking the inner nodes.

Finally, every parameter of the `SINUSOID` group has to be defined by a vector whose length corresponds to the number
of motors to drive.


### Ramps

They are ramp signals which increases of from an initial torque to a maximum torque, with a slope that ranges from `initial_speed` to `final_speed`.

Depending on the `bypass_joint_torque_measure` parameter, the initial torque is set to `0` (if the respective parameter is set to `true`) or to the measured one at the starting position (if the respective parameter is set to `false`).
While the maximum torque is given by the initial torque increased of `max_torque_increment`.

Note that, each ramp is repeated twice for each starting position, in order to cover both direction of motion.
Morever from one ramp to the following one the slope is incremented by `speed_increment`.

The sarting positions are computed like for the sinusoidal trajectories.

Finally, every parameter of the `RAMP` group has to be defined by a vector whose length corresponds to the number
of motors to drive.

### Examples

The following are some set of configurations which were tested on `ergoCubSN001`.

```
dt 0.002
safety_threshold 8.0
use_safety_threshold_for_starting_position false
number_of_starting_points 5
bypass_joint_torque_measure (true)
compensation_bias_forces_factor (0.0)

[SINUSOID]
min_delta_torque             ( 0.7 )
max_delta_torque             ( 1.8 )
delta_torque_increment       ( 0.2 )
min_frequency                 ( 0.05)
max_frequency                 ( 0.45)
frequency_decrement           ( 0.1 )

[RAMP]
initial_speed                    (0.15)
final_speed                      (0.40)
speed_increment                  (0.08)
max_torque_increment            (2.00)

[MOTOR]
joints_list         ("l_hip_roll", "l_hip_pitch", "l_hip_yaw",
                     "l_knee", "l_ankle_pitch", "l_ankle_roll",
                     "r_hip_roll", "r_hip_pitch", "r_hip_yaw",
                     "r_knee", "r_ankle_pitch", "r_ankle_roll")
k_tau                (0.094, 0.064, 0.150,
                     0.064, 0.064, 0.177,
                     0.094, 0.064, 0.150,
                     0.064, 0.064, 0.177)
max_safety_current  (8.0, 8.0, 4.0,
                     6.0, 3.5, 3.0,
                     8.0, 8.0, 4.0,
                     6.0, 3.5, 3.0)


robot_name                              ergocub

joints_list                             ("r_ankle_pitch")

remote_control_boards                   ("right_leg")
positioning_duration                    3.0  #in seconds
positioning_tolerance                   0.05 #in radians
position_direct_max_admissible_error    0.1  #in radians
```
