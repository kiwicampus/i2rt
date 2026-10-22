import enum
import logging
import os
import time
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface
from i2rt.motor_drivers.utils import MotorType

I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam.xml")
YAM_XML_LW_GRIPPER_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam_lw_gripper.xml")
YAM_XML_LINEAR_4310_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam_4310_linear.xml")
YAM_TEACHING_HANDLE_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam_teaching_handle.xml")
YAM_NO_GRIPPER_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam_no_gripper.xml")


class GripperType(enum.Enum):
    CRANK_4310 = "crank_4310"  # a 4310 motor with a crank
    LINEAR_3507 = "linear_3507"  # a 3507 motor with a linear actuator
    LINEAR_4310 = "linear_4310"  # a 4310 motor with a linear actuator

    # technically not a gripper
    YAM_TEACHING_HANDLE = "yam_teaching_handle"
    NO_GRIPPER = "no_gripper"

    @classmethod
    def from_string_name(cls, name: str) -> "GripperType":
        if name == "crank_4310":
            return cls.CRANK_4310
        elif name == "linear_3507":
            return cls.LINEAR_3507
        elif name == "linear_4310":
            return cls.LINEAR_4310
        elif name == "yam_teaching_handle":
            return cls.YAM_TEACHING_HANDLE
        elif name == "no_gripper":
            return cls.NO_GRIPPER
        else:
            raise ValueError(
                f"Unknown gripper type: {name}, gripper has to be one of the following: {GripperType.available_grippers()}"
            )

    @classmethod
    def available_grippers(cls) -> List[str]:
        return [gripper.value for gripper in GripperType]

    def get_gripper_limits(self) -> Optional[tuple[float, float]]:
        if self == GripperType.CRANK_4310:
            return 0.0, -2.7
        elif self in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]:
            return None
        elif self in [GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER]:
            return None

    def get_gripper_needs_calibration(self) -> bool:
        if self == GripperType.CRANK_4310:
            return True
        elif self in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]:
            return True
        elif self in [GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER]:
            return False

    def get_gripper_default_test_torque(self) -> float:
        if self == GripperType.CRANK_4310:
            return 0.4
        elif self == GripperType.LINEAR_4310:
            return 0.4
        elif self == GripperType.LINEAR_3507:
            return 0.5
        else:
            raise NotImplementedError

    def get_xml_path(self) -> str:
        if self == GripperType.CRANK_4310:
            return YAM_XML_PATH
        elif self == GripperType.LINEAR_3507:
            return YAM_XML_LW_GRIPPER_PATH
        elif self == GripperType.LINEAR_4310:
            return YAM_XML_LINEAR_4310_PATH
        elif self == GripperType.YAM_TEACHING_HANDLE:
            return YAM_TEACHING_HANDLE_PATH
        elif self == GripperType.NO_GRIPPER:
            return YAM_NO_GRIPPER_PATH
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_motor_kp_kd(self) -> tuple[float, float]:
        # POS_FORCE frames carry no gains: these apply in MIT before the mode switch, and kp == 0 means limp
        if self in [GripperType.CRANK_4310, GripperType.LINEAR_4310, GripperType.LINEAR_3507]:
            return 3.0, 0.15
        elif self == GripperType.YAM_TEACHING_HANDLE:
            return -1.0, -1.0  # no kp or kd for teaching handle
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_motor_type(self) -> str:
        if self in [GripperType.CRANK_4310, GripperType.LINEAR_4310]:
            return "DM4310"
        elif self in [GripperType.LINEAR_3507]:
            return "DM3507"
        elif self == GripperType.YAM_TEACHING_HANDLE:
            return ""  # or raise NotImplementedError
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_gripper_max_vel(self) -> float:
        """Gripper speed limit in normalized strokes per second."""
        return 4.5

    def get_gripper_move_torque(self) -> float:
        """Motor torque cap (Nm) while the gripper moves unblocked."""
        return 1.5

    def get_gripper_hold_torque(self) -> float:
        """Motor torque (Nm) while the gripper is blocked, friction included, as read back in the motor effort."""
        return 0.5

    def torque_to_current_pu(self, torque: float) -> float:
        """POS_FORCE current, per unit of the motor Imax, for a motor torque (Nm)."""
        torque_max = MotorType.get_motor_constants(self.get_motor_type()).TORQUE_MAX
        return float(np.clip(torque / torque_max, 0.0, 1.0))

    def get_gripper_clog_params(self) -> tuple[float, float, callable]:
        """
        clog_force_threshold: float,
        clog_speed_threshold: float,
        gripper_force_torque_map: callable,
        """
        if self == GripperType.CRANK_4310:
            return (
                0.3,
                0.2,
                partial(
                    zero_linkage_crank_gripper_force_torque_map,
                    motor_reading_to_crank_angle=lambda x: (-x + 0.174),
                    gripper_close_angle=8 / 180.0 * np.pi,
                    gripper_open_angle=170 / 180.0 * np.pi,
                    gripper_stroke=0.071,  # unit in meter
                ),
            )
        elif self in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]:
            return (
                0.3,
                0.3,
                partial(
                    linear_gripper_force_torque_map,
                    motor_stroke=6.57,
                    gripper_stroke=0.096,
                ),
            )
        elif self in [GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER]:
            return -1.0, -1.0, None


class JointMapper:
    def __init__(self, index_range_map: Dict[int, Tuple[float, float]], total_dofs: int):
        """_summary_
        This class is used to map the joint positions from the command space to the robot joint space.

        Args:
            index_range_map (Dict[int, Tuple[float, float]]): 0 indexed
            total_dofs (int): num of joints in the robot including the gripper if the girpper is the second robot
        """
        self.empty = len(index_range_map) == 0
        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs).astype(bool)
            self.joint_limits = []
            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))
            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_pos
        command_joint_pos = np.asarray(command_joint_pos, order="C")
        result = command_joint_pos.copy()
        needs_remapping = command_joint_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_vel
        result = command_joint_vel.copy()
        needs_remapping = command_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_vel
        result = robot_joint_vel.copy()
        needs_remapping = robot_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_pos
        result = robot_joint_pos.copy()
        needs_remapping = robot_joint_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result


def linear_gripper_force_torque_map(
    motor_stroke: float, gripper_stroke: float, gripper_force: float, current_angle: float
) -> float:
    """Maps the motor stroke required to achieve a given gripper force.

    Args:
        motor_stroke (float): in rad
        gripper_stroke (float): in meter
        gripper_force (float): in newton
    """
    # force = torque * motor_stroke / gripper_stroke
    return gripper_force * gripper_stroke / motor_stroke


def zero_linkage_crank_gripper_force_torque_map(
    gripper_close_angle: float,
    gripper_open_angle: float,
    motor_reading_to_crank_angle: Callable[[float], float],
    gripper_stroke: float,
    current_angle: float,
    gripper_force: float,
) -> float:
    """Maps the motor crank torque required to achieve a given gripper force. For Yam style gripper (zero linkage crank)

    Args:
        gripper_close_angle (float): Angle of the crank in radians at the closed position.
        gripper_open_angle (float): Angle of the crank in radians at the open position.
        gripper_stroke (float): Linear displacement of the gripper in meters.
        current_angle (float): Current crank angle in radians (relative to the closed position).
        gripper_force (float): Required gripping force in Newtons (N).

    Returns:
        float: Required motor torque in Newton-meters (Nm).
    """
    current_angle = motor_reading_to_crank_angle(current_angle)
    # Compute crank radius based on the total stroke and angle change
    crank_radius = gripper_stroke / (2 * (np.cos(gripper_close_angle) - np.cos(gripper_open_angle)))
    # gripper_position = crank_radius * (np.cos(gripper_close_angle) - np.cos(current_angle))
    grad_gripper_position = crank_radius * np.sin(current_angle)

    # Compute the required torque
    target_torque = gripper_force * grad_gripper_position
    return target_torque


class LockFreeCircularBuffer:
    """
    Lock-free circular buffer.
    There is a ~microsecond level race condition for this, but we're only using it to tell if the gripper is clogged or not.
    So 1 stale reading out of 1000 is not a big deal (FOR THAT PARTICULAR USE CASE!!!).
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.timestamps = np.zeros(maxsize)
        self.values = np.zeros(maxsize)
        self.write_idx = 0

    def put(self, timestamp: float, value: float) -> None:
        """Add a timestamped value to the buffer."""
        idx = self.write_idx % self.maxsize
        self.timestamps[idx] = timestamp
        self.values[idx] = value
        self.write_idx += 1

    def get_recent_values(self, time_window: float, current_time: Optional[float] = None) -> np.ndarray:
        """Get values within the specified time window."""
        if current_time is None:
            current_time = time.time()

        valid_mask = self.timestamps > (current_time - time_window)
        return self.values[valid_mask]


class GripperClogDetector:
    """Flags when the gripper is stalled against an object, from its averaged effort and speed."""

    def __init__(
        self,
        gripper_type: GripperType,
        average_torque_window: float = 0.05,  # in seconds
        clog_release_threshold: float = 0.15,  # in Nm
    ):
        self.gripper_type = gripper_type
        self.average_torque_window = average_torque_window
        self.clog_release_threshold = clog_release_threshold
        self.clog_force_threshold, self.clog_speed_threshold, _ = self.gripper_type.get_gripper_clog_params()
        self._is_clogged = False
        self._past_gripper_effort_buffer = LockFreeCircularBuffer(maxsize=1000)

    @property
    def is_clogged(self) -> bool:
        return self._is_clogged

    def update(self, gripper_state: Dict[str, float]) -> bool:
        self._past_gripper_effort_buffer.put(time.time(), gripper_state["current_eff"])
        relevant_history_effort = self._past_gripper_effort_buffer.get_recent_values(self.average_torque_window)
        if len(relevant_history_effort) > 0:
            average_effort = np.abs(np.mean(relevant_history_effort))
        else:
            average_effort = 0.0

        if self._is_clogged:
            normalized_current_qpos = gripper_state["current_normalized_qpos"]
            normalized_target_qpos = gripper_state["target_normalized_qpos"]
            # 0 close 1 open
            if normalized_current_qpos < normalized_target_qpos or average_effort < self.clog_release_threshold:
                self._is_clogged = False
        elif (
            average_effort > self.clog_force_threshold
            and np.abs(gripper_state["current_qvel"]) < self.clog_speed_threshold
        ):
            self._is_clogged = True
        return self._is_clogged


def detect_gripper_limits(
    motor_chain: DMChainCanInterface,
    gripper_index: int = 6,
    test_torque: float = 0.2,
    max_duration: float = 2.0,
    position_threshold: float = 0.01,
    check_interval: float = 0.1,
    open_inset: float = 0.01,
) -> List[float]:
    """
    Detect gripper limits by applying test torques and monitoring position changes.

    Args:
        motor_chain: Motor chain interface
        gripper_index: Index of gripper motor
        test_torque: Test torque for gripper detection (Nm)
        max_duration: Maximum test duration for each direction (s)
        position_threshold: Minimum position change to consider motor still moving (rad)
        check_interval: Time interval between checks (s)
        open_inset: Fraction of the stroke the open limit sits inside the stop, so position control rests unloaded

    Returns:
        List of detected limits [limit1, limit2]
    """
    logger = logging.getLogger(__name__)
    positions = []
    num_motors = len(motor_chain.motor_list)
    zero_torques = np.zeros(num_motors)

    # Get motor direction for the gripper
    motor_direction = motor_chain.motor_direction[gripper_index]

    # Record initial position
    initial_states = motor_chain.read_states()
    init_torque = np.array([state.eff for state in initial_states])
    initial_pos = initial_states[gripper_index].pos
    positions.append(initial_pos)
    logger.info(f"Gripper calibration starting from position: {initial_pos:.4f}")

    # Test both directions
    for direction in [1, -1]:
        logger.info(f"Testing gripper direction: {direction}")
        test_torques = init_torque
        test_torques[gripper_index] = direction * test_torque

        start_time = time.time()
        last_pos = None
        position_stable_count = 0

        while time.time() - start_time < max_duration:
            motor_chain.set_commands(torques=test_torques)
            time.sleep(check_interval)

            states = motor_chain.read_states()
            current_pos = states[gripper_index].pos
            positions.append(current_pos)

            # Check if position has stopped changing (gripper hit limit)
            if last_pos is not None:
                pos_change = abs(current_pos - last_pos)
                if pos_change < position_threshold:
                    position_stable_count += 1
                else:
                    position_stable_count = 0

                # Check if gripper has hit limit (position stable)
                if position_stable_count >= 6:  # tuned smaller to save time but less stable
                    logger.info(f"Gripper limit detected: pos={current_pos:.4f}")
                    break

            last_pos = current_pos

        time.sleep(0.3)

    # reset torque to zero, to prevent identifiy clogged
    motor_chain.set_commands(torques=np.array([0.0 for state in initial_states]))

    # Calculate detected limits
    min_pos = min(positions)
    max_pos = max(positions)

    # Order based on motor direction
    if motor_direction > 0:
        # Positive direction: [max, min]
        detected_limits = [max_pos, min_pos]
    else:
        # Negative direction: [min, max]
        detected_limits = [min_pos, max_pos]
    detected_limits[1] += open_inset * (detected_limits[0] - detected_limits[1])

    logger.info(f"Motor direction: {motor_direction}, detected limits: {detected_limits}")

    return detected_limits
