#!/usr/bin/env python3

# This software may be modified and distributed under the terms of the BSD-3-Clause license.

import datetime
import os
import signal
import sys
from abc import ABC, abstractmethod
from typing import Callable, Type
import threading

import bipedal_locomotion_framework as blf
import numpy as np
import yarp

import idyntree.bindings as iDynTree

logPrefix = "[PD-gravity-compensation-control]"

ParamHandler = Type[blf.parameters_handler.YarpParametersHandler]
RobotControl = Type[blf.robot_interface.YarpRobotControl]
SensorBridge = Type[blf.robot_interface.YarpSensorBridge]
PolyDriver = Type[blf.robot_interface.PolyDriver]

import threading

# Global variables to be modified interactively
interactive_params = {
    "joint_position_desired": None,
    "joint_velocity_desired": None,
    "Kp": None,
    "Kd": None
}
param_lock = threading.Lock()

def interactive_param_updater(joints_to_control): # TODO: use blf.info instead of print
    """
    Thread for interactive tuning using joint names.
    Commands:
      set pos <joint_name> <deg>
      set vel <joint_name> <deg/s>
      set kp  <joint_name> <Nm /deg>
      set kd  <joint_name> <Nm / (deg/s)>
      show
    """
    while True:
        try:
            cmd = input().strip().split()
            if not cmd:
                continue

            with param_lock:
                if cmd[0].lower() == "set" and len(cmd) == 3 + 1:  # set <type> <joint> <value>
                    param_type = cmd[1].lower()
                    joint_name = cmd[2]
                    value = float(cmd[3])

                    if joint_name not in joints_to_control:
                        blf.log().info(
                            f"[WARN] Joint '{joint_name}' is not in the list of controlled joints: {joints_to_control}"
                        )
                        continue

                    if param_type == "pos":
                        interactive_params["joint_position_desired"][joint_name] = np.deg2rad(value)
                    elif param_type == "vel":
                        interactive_params["joint_velocity_desired"][joint_name] = np.deg2rad(value)
                    elif param_type == "kp":
                        interactive_params["Kp"][joint_name] = value
                    elif param_type == "kd":
                        interactive_params["Kd"][joint_name] = value
                    else:
                        blf.log().info(
                            f"[WARN] Unknown parameter type '{param_type}'. Use 'pos', 'vel', 'kp', or 'kd'."
                        )
                
                elif cmd[0].lower() == "show":
                    blf.log().info("[INFO] Current parameters:")
                    for j in joints_to_control:
                        print(
                            f"{j}: pos={np.rad2deg(interactive_params['joint_position_desired'][j]):.2f} deg, "
                            f"vel={np.rad2deg(interactive_params['joint_velocity_desired'][j]):.2f} deg/s, "
                            f"Kp={interactive_params['Kp'][j]:.2f}, "
                            f"Kd={interactive_params['Kd'][j]:.2f}"
                        )
                else:
                    blf.log().info(
                        "[WARN] Invalid command. Use 'set <type> <joint> <value>' or 'show'."
                    )

        except Exception as e:
            blf.log().error(f"[ERROR] Exception in interactive parameter updater: {e}")
            blf.log().info("[INFO] Use 'show' to see current parameters or 'set' to modify them.")

class MotorParameters(ABC):
    # k_tau[A/Nm] includes the gear ratio
    k_tau = dict()
    # max_safety_current[A] limits the motor current to avoid damages
    max_safety_current = dict()

    @staticmethod
    def from_parameter_handler(motor_param_handler: ParamHandler) -> None:

        joints_list = motor_param_handler.get_parameter_vector_string("joints_list")
        k_tau = motor_param_handler.get_parameter_vector_float("k_tau")
        max_safety_current = motor_param_handler.get_parameter_vector_float(
            "max_safety_current"
        )

        if len(joints_list) != len(k_tau):
            raise ValueError(
                "{} The number of joints must be equal to the size of the k_tau".format(
                    logPrefix
                )
            )

        if len(joints_list) != len(max_safety_current):
            raise ValueError(
                "{} The number of joints must be equal to the size of the max_current".format(
                    logPrefix
                )
            )

        MotorParameters.k_tau = dict(zip(joints_list, k_tau))
        MotorParameters.max_safety_current = dict(zip(joints_list, max_safety_current))

class Trajectory(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

def build_remote_control_board_driver(
    param_handler: ParamHandler, local_prefix: str
) -> PolyDriver:
    param_handler.set_parameter_string("local_prefix", local_prefix)
    return blf.robot_interface.construct_remote_control_board_remapper(param_handler)

def create_ctrl_c_handler(
    sensor_bridge: SensorBridge, robot_control: RobotControl
) -> Callable[[int, object], None]:
    def ctrl_c_handler(sig, frame):
        blf.log().info("{} Ctrl+C pressed. Exiting gracefully.".format(logPrefix))
        # get the feedback
        if not sensor_bridge.advance():
            raise RuntimeError(
                "{} Unable to advance the sensor bridge".format(logPrefix)
            )

        are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()

        if not are_joints_ok:
            raise RuntimeError("{} Unable to get the joint positions".format(logPrefix))

        # set the control mode to position
        if not robot_control.set_control_mode(
            blf.robot_interface.YarpRobotControl.Position
        ):
            raise RuntimeError("{} Unable to set the control mode".format(logPrefix))
        if not robot_control.set_references(
            joint_positions, blf.robot_interface.YarpRobotControl.Position
        ):
            raise RuntimeError("{} Unable to set the references".format(logPrefix))

        blf.log().info(
            "{} Sleep for two seconds. Just to be sure the interfaces are on.".format(
                logPrefix
            )
        )
        blf.clock().sleep_for(datetime.timedelta(seconds=2))
        sys.exit(0)

    return ctrl_c_handler

def compute_bias_forces(
    dynComp: iDynTree.KinDynComputations,
    sensor_bridge: SensorBridge,
    generalizedBiasForcesVector: iDynTree.FreeFloatingGeneralizedTorques
    ) -> bool:

    """ This function computes the generalized bias forces using the iDynTree library.
    It retrieves the necessary quantities from the sensor bridge and sets the robot state in the iDynTree computations object.
    Finally, it computes the generalized bias forces and stores them in the provided vector.
    Args:
        dynComp (iDynTree.KinDynComputations): The iDynTree computations object
        sensor_bridge (SensorBridge): The sensor bridge to retrieve the necessary quantities
        generalizedBiasForcesVector (iDynTree.FreeFloatingGeneralizedTorques): The vector to store the computed generalized bias forces
    Returns:
        bool: True if the computation was successful, False otherwise
    """

    # We need some dynamic quantities to compute the bias forces

    # 1) homogeneous transformation matrix from the base to the world
    base_link = "root_link"   # TODO avoid hardocoding
    index_base = dynComp.getFrameIndex(base_link)
    world_T_base = dynComp.getWorldTransform(index_base)

    # 2) joint positions
    are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()
    if not are_joints_ok:
        raise RuntimeError("Could not get joint positions")

    # 3) base velocity
    base_velocity = 6 * [0.0]

    # 4) joint velocities
    are_joints_ok, joint_velocities, _ = sensor_bridge.get_joint_velocities()

    # 5) gravity vector
    g = [0.0, 0.0, -9.81]

    # Now we can set the robot state
    if (not dynComp.setRobotState(world_T_base, joint_positions, base_velocity, joint_velocities, g)):
        raise RuntimeError("Could not set robot state")

    # Finally, we can compute the generalized bias forces
    if (not dynComp.generalizedBiasForces(generalizedBiasForcesVector)):
        raise RuntimeError("Could not compute generalized bias forces")
    
    return True

def main():

    # ========== PARAMETER LOADING ==========
    # Load configuration parameters from INI file
    param_handler = blf.parameters_handler.YarpParametersHandler()
    param_file = "blf-PD-gravity-compensation-control-options.ini"

    if not param_handler.set_from_filename(param_file):
        raise RuntimeError("{} Unable to load the parameters".format(logPrefix))

    # Load desired joint positions and velocities (control targets)
    joint_position_desired = param_handler.get_parameter_vector_float(
        "joint_position_desired"
    )
    joint_velocity_desired = param_handler.get_parameter_vector_float(
        "joint_velocity_desired"
    )

    # Convert desired joint positions and velocities to degrees
    joint_position_desired = np.deg2rad(joint_position_desired)
    joint_velocity_desired = np.deg2rad(joint_velocity_desired)

    # Get the PD gains from the configuration file
    Kp = param_handler.get_parameter_vector_float("Kp")
    Kd = param_handler.get_parameter_vector_float("Kd")

    # Load the motor parameters
    MotorParameters.from_parameter_handler(param_handler.get_group("MOTOR"))

    # ========== ROBOT INTERFACE SETUP ==========
    # Load list of joints to be controlled
    robot_control_handler = param_handler.get_group("ROBOT_CONTROL")
    joints_to_control = robot_control_handler.get_parameter_vector_string("joints_list")
    blf.log().info("{} Joints to control: {}".format(logPrefix, joints_to_control))

    # Verify that all joints to control have motor parameters defined
    for joint in joints_to_control:
        if joint not in MotorParameters.max_safety_current.keys():
            raise RuntimeError(
                "{} The joint {} is not supported by the application".format(
                    logPrefix, joint
                )
            )
    
    # Store initial parameters in dicts keyed by joint name
    with param_lock:
        interactive_params["joint_position_desired"] = dict(zip(
            joints_to_control, joint_position_desired
        ))
        interactive_params["joint_velocity_desired"] = dict(zip(
            joints_to_control, joint_velocity_desired
        ))
        interactive_params["Kp"] = dict(zip(
            joints_to_control, Kp
        ))
        interactive_params["Kd"] = dict(zip(
            joints_to_control, Kd
        ))

    # Start input thread
    input_thread = threading.Thread(
        target=interactive_param_updater,
        args=(joints_to_control,),
        daemon=True
    )
    input_thread.start()
    
    # Load compensation factors for bias forces (allows scaling gravity compensation)
    compensation_bias_forces_factor = param_handler.get_parameter_vector_float(
        "compensation_bias_forces_factor"
        )

    # Validate compensation factors array size
    if len(compensation_bias_forces_factor) != len(joints_to_control):
        raise ValueError(
            "{} The number of joints must be equal to the size of the compensation_bias_forces_factor parameter".format(
                logPrefix
            )
        )

    # Initialize YARP drivers for communication with robot
    poly_drivers = dict()

    # Create remote control board driver for sending commands
    poly_drivers["REMOTE_CONTROL_BOARD"] = build_remote_control_board_driver(
        param_handler=robot_control_handler,
        local_prefix="PD-gravity-compensation-control",
    )
    if not poly_drivers["REMOTE_CONTROL_BOARD"].is_valid():
        raise RuntimeError(
            "{} Unable to create the remote control board driver".format(logPrefix)
        )

    # Wait for interfaces to be ready
    blf.log().info(
        "{} Sleep for two seconds. Just to be sure the interfaces are on.".format(
            logPrefix
        )
    )
    blf.clock().sleep_for(datetime.timedelta(seconds=2))

    # Initialize robot control interface
    robot_control = blf.robot_interface.YarpRobotControl()
    if not robot_control.initialize(robot_control_handler):
        raise RuntimeError(
            "{} Unable to initialize the robot control".format(logPrefix)
        )
    if not robot_control.set_driver(poly_drivers["REMOTE_CONTROL_BOARD"].poly):
        raise RuntimeError(
            "{} Unable to set the driver for the robot control".format(logPrefix)
        )

    # ========== SAFETY LIMITS SETUP ==========
    # Load safety threshold for joint limits
    safety_threshold = param_handler.get_parameter_float("safety_threshold")
    safety_threshold = np.deg2rad(safety_threshold)

    # Get hardware joint limits from robot
    _, lower_limits, upper_limits = robot_control.get_joint_limits()
    lower_limits = np.deg2rad(lower_limits)
    upper_limits = np.deg2rad(upper_limits)

    # Check for software-defined joint limits (more restrictive than hardware limits)
    try:
        software_lower_limits = param_handler.get_parameter_vector_float(
            "software_joint_lower_limits"
        )
        software_upper_limits = param_handler.get_parameter_vector_float(
            "software_joint_upper_limits"
        )
    except ValueError:
        software_lower_limits = None
        software_upper_limits = None

    # Apply software limits if they exist (use most restrictive limits)
    if software_lower_limits is not None and software_upper_limits is not None:
        if len(software_lower_limits) != len(joints_to_control) or len(software_upper_limits) != len(joints_to_control):
            raise ValueError(
                "{} The number of joints must be equal to the size of the software joint limits".format(
                    logPrefix
                )
            )
        # Use maximum of hardware lower limit and software lower limit
        lower_limits = np.maximum(lower_limits, np.deg2rad(software_lower_limits))
        # Use minimum of hardware upper limit and software upper limit
        upper_limits = np.minimum(upper_limits, np.deg2rad(software_upper_limits))

    # ========== SENSOR BRIDGE SETUP ==========
    # Create sensor bridge for reading robot state
    sensor_bridge = blf.robot_interface.YarpSensorBridge()
    sensor_bridge_handler = param_handler.get_group("SENSOR_BRIDGE")
    if not sensor_bridge.initialize(sensor_bridge_handler):
        raise RuntimeError(
            "{} Unable to initialize the sensor bridge".format(logPrefix)
        )
    if not sensor_bridge.set_drivers_list(list(poly_drivers.values())):
        raise RuntimeError(
            "{} Unable to set the drivers for the sensor bridge".format(logPrefix)
        )
    if not sensor_bridge.advance():
        raise RuntimeError("{} Unable to advance the sensor bridge".format(logPrefix))

    # Get initial joint positions and velocities
    are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()
    if not are_joints_ok:
        raise RuntimeError("{} Unable to get the joint positions".format(logPrefix))
    
    are_joints_ok, joint_velocities, _ = sensor_bridge.get_joint_velocities()
    if not are_joints_ok:
        raise RuntimeError("{} Unable to get the joint velocities".format(logPrefix))

    # ========== DYNAMICS COMPUTATION SETUP ==========
    # Load robot model for dynamics computation (gravity compensation)
    URDF_FILE = os.path.join(os.getenv('ROBOTOLOGY_SUPERBUILD_INSTALL_PREFIX'), 'share/ergoCub/robots', os.getenv('YARP_ROBOT_NAME'), 'model.urdf')
    dynComp = iDynTree.KinDynComputations()
    mdlLoader = iDynTree.ModelLoader()

    # Load reduced model containing only the joints we want to control
    if (not mdlLoader.loadReducedModelFromFile(URDF_FILE, joints_to_control)):
        raise RuntimeError("Could not load model from file: " + URDF_FILE)
    dynComp.loadRobotModel(mdlLoader.model())
    
    # ========== DATA LOGGING SETUP ==========
    # Create server for logging control data
    vectors_collection_server = blf.yarp_utilities.VectorsCollectionServer()
    if not vectors_collection_server.initialize(
        param_handler.get_group("DATA_LOGGING")
    ):
        raise RuntimeError(
            "{} Unable to initialize the vectors collection server".format(logPrefix)
        )

    vectors_collection_server.populate_metadata(
        "joints::desired::torque",
        joints_to_control,
    )
    vectors_collection_server.populate_metadata(
        "joints::desired::position",
        joints_to_control,
    )
    vectors_collection_server.populate_metadata(
        "joints::desired::velocity",
        joints_to_control,
    )
    vectors_collection_server.populate_metadata(
        "joints::gains::Kp",
        joints_to_control,
    )
    vectors_collection_server.populate_metadata(
        "joints::gains::Kd",
        joints_to_control,
    )
    vectors_collection_server.populate_metadata(
        "joints::control_mode::torque",
        joints_to_control,
    )
    vectors_collection_server.finalize_metadata()
    blf.clock().sleep_for(datetime.timedelta(milliseconds=200))

    # ========== SIGNAL HANDLER SETUP ==========
    # Set up Ctrl+C handler for graceful shutdown
    ctrl_c_handler = create_ctrl_c_handler(
        sensor_bridge=sensor_bridge, robot_control=robot_control
    )
    signal.signal(signal.SIGINT, ctrl_c_handler)

    # ========== CONTROL LOOP SETUP ==========
    # Set control loop time step
    dt = param_handler.get_parameter_datetime("dt")

    # Wait for user input before starting control
    blf.log().info(
        "{} Waiting for your input, press ENTER to start the data collection".format(
            logPrefix
        )
    )
    input()
    blf.log().info("{} Start".format(logPrefix))

    # Initialize iDynTree vector for storing computed bias forces
    generalizedBiasForcesVector_idyn = iDynTree.FreeFloatingGeneralizedTorques(mdlLoader.model())  

    # Initialize error vectors for PD control
    position_error = np.zeros(len(joints_to_control))
    velocity_error = np.zeros(len(joints_to_control))

    # Initalize reference vectors
    joint_torque_reference = np.zeros(len(joints_to_control))
    current_reference = np.zeros(len(joints_to_control))

    # Set control modes: Torque for real robot and Gazebo for each joint
    control_modes = [blf.robot_interface.YarpRobotControl.Torque for _ in joints_to_control]

    # Update control modes for all joints
    if not robot_control.set_control_mode(control_modes):
        raise RuntimeError("{} Unable to set the control mode".format(logPrefix))

    # ========== MAIN CONTROL LOOP ==========
    ROTTO = False
    try:
        while True:
            # Record start time for timing control
            tic = blf.clock().now()

            # Initialize is_out_of_safety_limits ONCE per control cycle
            is_out_of_safety_limits = [False for _ in joints_to_control]

            # ========== READ SENSOR DATA ==========
            # Get the measured joint positions
            if not sensor_bridge.advance():
                raise RuntimeError(
                    "{} Unable to advance the sensor bridge".format(logPrefix)
                )
            are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()
            if not are_joints_ok:
                raise RuntimeError("{} Unable to get the joint positions".format(logPrefix))
            # Get the measured joint velocities
            are_joints_ok, joint_velocities, _ = sensor_bridge.get_joint_velocities()
            if not are_joints_ok:
                raise RuntimeError("{} Unable to get the joint velocities".format(logPrefix))
            
            # ========== GRAVITY COMPENSATION COMPUTATION ==========
            # Compute bias forces (gravity + Coriolis + centrifugal forces)
            if not compute_bias_forces(dynComp, sensor_bridge, generalizedBiasForcesVector_idyn):
                raise RuntimeError("{} Unable to compute the bias forces".format(logPrefix))
            # Extract joint torques from generalized bias forces
            generalizedBiasForcesVector = generalizedBiasForcesVector_idyn.jointTorques().toNumPy()
            
            # ========== SAFETY CHECK AND CONTROL COMPUTATION ==========
            for joint_idx, joint_name in enumerate(joints_to_control):
                # Check if the joint is within the safety limits
                if (
                    joint_positions[joint_idx] < lower_limits[joint_idx] + safety_threshold
                    or joint_positions[joint_idx] > upper_limits[joint_idx] - safety_threshold
                ):
                    # set the control mode to position
                    control_modes[
                        joint_idx
                    ] = blf.robot_interface.YarpRobotControl.Position

                    # set the reference to the current position
                    joint_position_desired[joint_idx] = joint_positions[joint_idx]

                    # update flag
                    is_out_of_safety_limits[joint_idx] = True

                    # set control modes
                    if not robot_control.set_control_mode(control_modes):
                        raise RuntimeError(
                            "{} Unable to set the control mode".format(logPrefix)
                        )

                    blf.log().warn(
                        "{} Joint {} is out of the safety limits, stopping its trajectory and switching to Position control with reference position {}".format(
                            logPrefix, joint, joint_position_desired[joint_idx]
                        )
                    )
                
                # Update parameters from interactive input
                with param_lock:
                    joint_position_desired = np.array([interactive_params["joint_position_desired"][j] for j in joints_to_control])
                    joint_velocity_desired = np.array([interactive_params["joint_velocity_desired"][j] for j in joints_to_control])
                    Kp = np.array([interactive_params["Kp"][j] for j in joints_to_control])
                    Kd = np.array([interactive_params["Kd"][j] for j in joints_to_control])
                    
                # ========== PD CONTROL COMPUTATION ==========
                # Compute position error (desired - actual)
                position_error[joint_idx] = joint_position_desired[joint_idx] - joint_positions[joint_idx]
                # Compute velocity error (desired - actual)
                velocity_error[joint_idx] = joint_velocity_desired[joint_idx] - joint_velocities[joint_idx]

                # Scale the bias forces by the compensation factor for each joint (TODO: check if this is needed)
                generalizedBiasForcesVector[joint_idx]*= compensation_bias_forces_factor[joint_idx]

                # ========== TOTAL TORQUE COMPUTATION ==========
                # PD control torque + gravity compensation torque
                joint_torque_reference[joint_idx] = Kp[joint_idx] * position_error[joint_idx] + Kd[joint_idx] * velocity_error[joint_idx] + generalizedBiasForcesVector[joint_idx]

                # ========== TORQUE TO CURRENT CONVERSION ==========
                # Convert desired torque to motor current using motor torque constant
                current_reference[joint_idx] = joint_torque_reference[joint_idx] * MotorParameters.k_tau[joints_to_control[joint_idx]]

                # ========== CURRENT SAFETY CHECK ==========
                # Check if computed current exceeds motor safety limits
                if (
                    np.abs(current_reference[joint_idx])
                    > MotorParameters.max_safety_current[
                        joints_to_control[joint_idx]
                    ]
                    ):
                        raise RuntimeError(
                            "{} The current reference for joint {} is exceeding the safety limits, exiting the application".format(
                                logPrefix, joints_to_control[joint_idx]
                            )
                        )

            # ========== REFERENCE SELECTION ==========
            # Choose between position reference (when joint pos is out of safety limits) or torque reference
            reference = np.where(
                is_out_of_safety_limits, joint_positions, np.array(joint_torque_reference)
            )

            # ========== SEND COMMANDS TO ROBOT ==========
            # Send computed references to robot
            if not robot_control.set_references(
                reference, control_modes, joint_positions
            ):
                raise RuntimeError("{} Unable to set the references".format(logPrefix))

            # ========== SAFETY LIMITS CHECK ==========
            # Check if all joints are out of safety limits
            if all(is_out_of_safety_limits):
                blf.log().info(
                    "{} The torque control is stopped due to all joints exceeding safety limits. Going back to position control.".format(
                        logPrefix
                    )
                )
                break

            # ========== DATA LOGGING ==========
            # Prepare and send logging data
            vectors_collection_server.prepare_data()
            if not vectors_collection_server.clear_data():
                raise RuntimeError("{} Unable to clear the data".format(logPrefix))
            vectors_collection_server.populate_data(
                "joints::desired::torque", np.array(joint_torque_reference)
            )
            vectors_collection_server.populate_data(
                "joints::desired::position", np.array(joint_position_desired)
            )
            vectors_collection_server.populate_data(
                "joints::desired::velocity", np.array(joint_velocity_desired)
            )
            vectors_collection_server.populate_data(
                "joints::gains::Kp", np.array(Kp)
            )
            vectors_collection_server.populate_data(
                "joints::gains::Kd", np.array(Kd)
            )
            vectors_collection_server.populate_data(
                "joints::control_mode::torque", np.where(is_out_of_safety_limits, 0, 1)
            )
            vectors_collection_server.send_data()

            # ========== TIMING CONTROL ==========
            # Ensure control loop runs at specified frequency
            toc = blf.clock().now()
            delta_time = toc - tic
            if delta_time < dt:
                blf.clock().sleep_for(dt - delta_time)
            else:
                blf.log().debug(
                    "{} The control loop is too slow, real time constraints not satisfied".format(
                        logPrefix
                    )
                )
        
    except Exception as e:
        blf.log().error(f"{logPrefix} Exception occurred: {e}")
        ctrl_c_handler(None, None)  # Safe exit
    
    # # get the feedback
    # if not sensor_bridge.advance():
    #     raise RuntimeError("{} Unable to advance the sensor bridge".format(logPrefix))

    # are_joints_ok, joint_positions, _ = sensor_bridge.get_joint_positions()

    # if not are_joints_ok:
    #     raise RuntimeError("{} Unable to get the joint positions".format(logPrefix))

    # # set the control mode to position
    # if not robot_control.set_control_mode(
    #     blf.robot_interface.YarpRobotControl.Position
    # ):
    #     raise RuntimeError("{} Unable to set the control mode".format(logPrefix))
    # if not robot_control.set_references(
    #     joint_positions, blf.robot_interface.YarpRobotControl.Position
    # ):
    #     raise RuntimeError("{} Unable to set the references".format(logPrefix))


if __name__ == "__main__":
    network = yarp.Network()
    main()
