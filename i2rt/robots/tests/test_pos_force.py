import struct
import threading
from types import SimpleNamespace
from typing import Iterator, List
from unittest import mock

import can
import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import (
    ControlMode,
    DMChainCanInterface,
    DMSingleMotorCanInterface,
    MotorCmd,
    PosForceLimits,
)
from i2rt.motor_drivers.utils import ReceiveMode
from i2rt.robots import motor_chain_robot, utils
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import (
    GRIPPER_CLOG_FORCE,
    GRIPPER_CLOG_RELEASE_FORCE,
    GripperClogDetector,
    GripperType,
    detect_gripper_limits,
)

MOTOR_ID = 0x07
RECV_ID = MOTOR_ID + 16


# ----------------- driver ----------------- #


def make_interface() -> DMSingleMotorCanInterface:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    iface.control_mode = ControlMode.MIT
    iface.cmd_idoffset = 0
    iface.receive_mode = ReceiveMode.p16
    iface.name = "test"
    iface.bus = SimpleNamespace(channel_info="vcan")
    return iface


def feedback_message() -> can.Message:
    # error 0x1, mid range position, velocity and torque
    return can.Message(arbitration_id=RECV_ID, data=[0x17, 0x80, 0x00, 0x80, 0x08, 0x00, 30, 30])


def param_reply(op: int, rid: int, value: int) -> can.Message:
    return can.Message(arbitration_id=RECV_ID, data=[MOTOR_ID, 0, op, rid] + list(struct.pack("<I", value)))


def test_pos_force_frame_layout() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(return_value=feedback_message())
    iface.set_control(MOTOR_ID, "DM4310", 1.5, 0, 0, 0, 0, ControlMode.POS_FORCE, vel_limit=26.0, current_limit=0.0592)
    frame_id, motor_id, data = iface._send_message_get_response.call_args.args
    assert frame_id == 0x300 + MOTOR_ID
    pos, v_des, i_des = struct.unpack("<fHH", bytes(data))
    assert pos == pytest.approx(1.5)
    assert v_des == 2600
    assert i_des == 592


def test_pos_force_frame_clamps_limits() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(return_value=feedback_message())
    iface.set_control(MOTOR_ID, "DM4310", 0, 0, 0, 0, 0, ControlMode.POS_FORCE, vel_limit=500, current_limit=2)
    _, v_des, i_des = struct.unpack("<fHH", bytes(iface._send_message_get_response.call_args.args[2]))
    assert (v_des, i_des) == (10000, 10000)


def test_mit_frame_unchanged_by_default() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(return_value=feedback_message())
    iface.set_control(MOTOR_ID, "DM4310", 0, 0, 3.0, 0.15, 0)
    assert iface._send_message_get_response.call_args.args[0] == MOTOR_ID


def test_set_control_mode_writes_and_verifies() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(side_effect=[param_reply(0x55, 10, 4), param_reply(0x33, 10, 4)])
    iface.set_control_mode(MOTOR_ID, ControlMode.POS_FORCE)
    write, read = iface._send_message_get_response.call_args_list
    assert write.args == (0x7FF, MOTOR_ID, [MOTOR_ID, 0, 0x55, 10, 4, 0, 0, 0])
    assert read.args == (0x7FF, MOTOR_ID, [MOTOR_ID, 0, 0x33, 10, 0, 0, 0, 0])


def test_set_control_mode_raises_when_rejected() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(side_effect=[param_reply(0x55, 10, 4), param_reply(0x33, 10, 1)])
    with pytest.raises(RuntimeError, match="rejected"):
        iface.set_control_mode(MOTOR_ID, ControlMode.POS_FORCE)


def test_set_control_mode_raises_on_feedback_reply() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock(side_effect=[param_reply(0x55, 10, 4), feedback_message()])
    with pytest.raises(RuntimeError, match="unexpected reply"):
        iface.set_control_mode(MOTOR_ID, ControlMode.POS_FORCE)


def test_float_register_write_packs_little_endian() -> None:
    iface = make_interface()
    iface._send_message_get_response = mock.Mock()
    iface.write_param_float(MOTOR_ID, 27, 73.0)
    assert iface._send_message_get_response.call_args.args == (
        0x7FF,
        MOTOR_ID,
        [MOTOR_ID, 0, 0x55, 27] + list(struct.pack("<f", 73.0)),
    )


def make_chain() -> DMChainCanInterface:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_list = [(MOTOR_ID, "DM4310")]
    chain.motor_offset = np.array([0.0])
    chain.motor_direction = np.array([1])
    chain.control_modes = [ControlMode.POS_FORCE]
    chain.pos_force_limits = [PosForceLimits(vel=24.0, current=0.05)]
    chain.absolute_positions = np.array([2.0])
    chain.command_lock = threading.Lock()
    chain.state_lock = threading.Lock()
    chain.motor_interface = mock.Mock()
    return chain


def test_chain_zero_kp_sends_zero_current() -> None:
    chain = make_chain()
    chain._set_commands([MotorCmd(pos=1.0, kp=0.0)])
    assert chain.motor_interface.set_control.call_args.kwargs["current_limit"] == 0.0
    chain._set_commands([MotorCmd(pos=1.0, kp=3.0)])
    kwargs = chain.motor_interface.set_control.call_args.kwargs
    assert (kwargs["control_mode"], kwargs["vel_limit"], kwargs["current_limit"]) == (
        ControlMode.POS_FORCE,
        24.0,
        0.05,
    )


def test_chain_runtime_switch_holds_position() -> None:
    chain = make_chain()
    chain.control_modes = [ControlMode.MIT]
    chain.commands = [MotorCmd(pos=0.0, kp=3.0, kd=0.15)]
    chain.motor_interface.try_receive_message.return_value = None
    chain.set_control_mode(0, ControlMode.POS_FORCE)
    calls = [c[0] for c in chain.motor_interface.method_calls if c[0] != "try_receive_message"]
    assert calls == ["motor_off", "set_control_mode", "motor_on"]
    assert chain.control_modes == [ControlMode.POS_FORCE]
    assert chain.commands[0] == MotorCmd(pos=2.0, kp=3.0, kd=0.15)


def test_chain_register_float_verifies_read_back() -> None:
    chain = make_chain()
    chain.motor_interface.read_param_float.return_value = 73.0
    chain.set_register_float(0, 27, 73.0)
    chain.motor_interface.write_param_float.assert_called_once_with(MOTOR_ID, 27, 73.0)
    chain.motor_interface.read_param_float.return_value = 54.0
    with pytest.raises(RuntimeError, match="rejected RID 27"):
        chain.set_register_float(0, 27, 73.0)


def test_chain_limits_are_clipped() -> None:
    chain = make_chain()
    chain.set_pos_force_limits(0, vel=10.0, current=3.0)
    assert chain.pos_force_limits[0] == PosForceLimits(vel=10.0, current=1.0)


# ----------------- clog detector ----------------- #


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(utils.time, "time", fake)
    return fake


def gripper_state(eff: float, vel: float = 0.0, current: float = 0.4, target: float = 0.0, qpos: float = 3.0) -> dict:
    return {
        "current_qpos": qpos,
        "current_qvel": vel,
        "current_eff": eff,
        "current_normalized_qpos": current,
        "target_normalized_qpos": target,
    }


def feed(detector: GripperClogDetector, clock: FakeClock, eff: float, duration: float, **kwargs: float) -> bool:
    for _ in range(round(duration / 0.003)):
        clock.t += 0.003
        clogged = detector.update(gripper_state(eff, **kwargs))
    return clogged


def test_gripper_type_values() -> None:
    for gripper_type in [GripperType.CRANK_4310, GripperType.LINEAR_4310, GripperType.LINEAR_3507]:
        assert gripper_type.get_motor_kp_kd() == (3.0, 0.15)
        assert gripper_type.get_gripper_max_vel() == 4.5
        assert gripper_type.get_gripper_move_torque() == 1.5
        assert gripper_type.get_gripper_hold_force() == 30.0
        detector = GripperClogDetector(gripper_type)
        assert detector.clog_force_threshold == 20.0
        assert detector.clog_release_threshold == 10.0
        assert detector.average_torque_window == 0.05


# linear gripper: 0.096 m of jaw stroke over 6.57 rad of motor travel
LINEAR_NM_PER_N = 0.096 / 6.57


def test_linear_thresholds_match_effort_in_nm() -> None:
    for effort in (0.29, 0.15):
        force = GripperType.LINEAR_4310.effort_to_force(effort, current_angle=3.0)
        assert force == pytest.approx(effort / LINEAR_NM_PER_N)
    assert GRIPPER_CLOG_FORCE * LINEAR_NM_PER_N == pytest.approx(0.29, abs=0.005)
    assert GRIPPER_CLOG_RELEASE_FORCE * LINEAR_NM_PER_N == pytest.approx(0.15, abs=0.005)


def test_linear_hold_torque_is_position_independent() -> None:
    expected = 0.3 + 30.0 * LINEAR_NM_PER_N
    for gripper_type in (GripperType.LINEAR_4310, GripperType.LINEAR_3507):
        for angle in (0.0, 3.0, 6.0):
            assert gripper_type.force_to_torque(30.0, current_angle=angle) == pytest.approx(expected)
    assert GripperType.LINEAR_4310.force_to_current_pu(30.0, current_angle=0.0) == pytest.approx(expected / 10.0)


def crank_nm_per_n(raw: float) -> float:
    radius = 0.071 / (2 * (np.cos(np.deg2rad(8)) - np.cos(np.deg2rad(170))))
    return radius * np.sin(-raw + 0.174)


def test_crank_hold_torque_follows_position() -> None:
    torques = {raw: GripperType.CRANK_4310.force_to_torque(30.0, current_angle=raw) for raw in (0.0, -1.4, -2.7)}
    for raw, torque in torques.items():
        assert torque == pytest.approx(0.3 + 30.0 * crank_nm_per_n(raw))
    # the crank needs the most torque mid stroke, where its lever arm is longest
    assert torques[-1.4] > torques[0.0] and torques[-1.4] > torques[-2.7]


def test_effort_to_force_inverts_the_frictionless_map() -> None:
    for raw in (0.0, -1.4, -2.7):
        force = GripperType.CRANK_4310.effort_to_force(0.5, current_angle=raw)
        assert force == pytest.approx(0.5 / crank_nm_per_n(raw))


def test_latches_above_threshold_when_slow(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    assert feed(detector, clock, eff=0.35, duration=0.06)
    assert detector.is_clogged


def test_no_latch_below_threshold(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    assert not feed(detector, clock, eff=0.25, duration=0.2)


def test_no_latch_while_moving(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    assert not feed(detector, clock, eff=1.0, duration=0.06, vel=0.5)


def test_window_drops_old_samples(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    feed(detector, clock, eff=1.0, duration=0.03, vel=0.5)
    feed(detector, clock, eff=0.1, duration=0.06, vel=0.5)
    # only the 0.1 samples are inside the 0.05 s window
    assert not feed(detector, clock, eff=0.1, duration=0.003)


def test_releases_below_release_threshold(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    assert feed(detector, clock, eff=0.35, duration=0.06)
    assert feed(detector, clock, eff=0.18, duration=0.06)
    assert not feed(detector, clock, eff=0.1, duration=0.06)


def test_releases_on_open_command(clock: FakeClock) -> None:
    detector = GripperClogDetector(GripperType.LINEAR_4310)
    assert feed(detector, clock, eff=0.35, duration=0.06)
    assert not feed(detector, clock, eff=0.35, duration=0.003, target=1.0)


def test_crank_stays_clogged_at_hold_torque_near_closed(clock: FakeClock) -> None:
    # near closed the 30 N hold needs little torque; the force thresholds keep it latched
    detector = GripperClogDetector(GripperType.CRANK_4310)
    hold = GripperType.CRANK_4310.force_to_torque(30.0, current_angle=-0.05)
    assert feed(detector, clock, eff=0.6, duration=0.06, qpos=-0.05)
    assert feed(detector, clock, eff=hold, duration=0.2, qpos=-0.05)


def test_torque_to_current_uses_torque_max() -> None:
    # DM4310 peaks at 10 Nm, DM3507 at 5 Nm
    assert GripperType.LINEAR_4310.torque_to_current_pu(0.5) == pytest.approx(0.05)
    assert GripperType.LINEAR_3507.torque_to_current_pu(0.5) == pytest.approx(0.1)
    assert GripperType.LINEAR_4310.torque_to_current_pu(20.0) == 1.0


def run_detect_gripper_limits(first_sweep: list, second_sweep: list, direction: int, **kwargs: float) -> list:
    positions = iter([0.0] + first_sweep + [first_sweep[-1]] * 10 + second_sweep + [second_sweep[-1]] * 10)

    def read_states() -> list:
        pos = next(positions)
        return [SimpleNamespace(pos=0.0, eff=0.0), SimpleNamespace(pos=pos, eff=0.0)]

    chain = SimpleNamespace(
        motor_list=[0, 1],
        motor_direction=np.array([1, direction]),
        read_states=read_states,
        set_commands=mock.Mock(),
    )
    with mock.patch.object(utils.time, "sleep"):
        return detect_gripper_limits(chain, gripper_index=1, check_interval=0.0, max_duration=1e9, **kwargs)


def test_detect_gripper_limits_insets_only_open() -> None:
    # closed stays on the stop; open moves 1% of the 1.05 stroke toward closed
    limits = run_detect_gripper_limits([-0.5, -1.0], [0.0, 0.05], direction=1)
    assert limits == pytest.approx([0.05, -1.0 + 0.0105])


def test_detect_gripper_limits_inset_follows_negative_direction() -> None:
    limits = run_detect_gripper_limits([0.5, 2.0], [0.0, -1.0], direction=-1)
    assert limits == pytest.approx([-1.0, 2.0 - 0.03])


def test_detect_gripper_limits_zero_inset_keeps_stops() -> None:
    limits = run_detect_gripper_limits([-0.5, -1.0], [0.0, 0.05], direction=1, open_inset=0.0)
    assert limits == [0.05, -1.0]


# ----------------- robot ----------------- #

# raw gripper limits: 0.0 is closed (normalized 0), 6.0 is open (normalized 1)
LIMITS = np.array([0.0, 6.0])


class FakeChain:
    """Arm joint 0 and gripper 1; the gripper reports whatever position and effort the test sets."""

    def __init__(self, reject: bool = False):
        self.motor_list = [(1, "DM4340"), (MOTOR_ID, "DM4310")]
        self.start_thread_flag = True
        self.running = True
        self.gripper_pos = 3.0
        self.gripper_eff = 0.0
        self.last_pos: np.ndarray = np.zeros(2)
        self.set_pos_force_limits = mock.Mock()
        self.set_control_mode = mock.Mock(side_effect=RuntimeError("rejected") if reject else None)
        self.set_register_float = mock.Mock()

    def __len__(self) -> int:
        return 2

    def read_states(self) -> List[SimpleNamespace]:
        return [
            SimpleNamespace(pos=0.0, vel=0.0, eff=0.0, temp_mos=0.0, temp_rotor=0.0, timestamp=0.0),
            SimpleNamespace(
                pos=self.gripper_pos, vel=0.0, eff=self.gripper_eff, temp_mos=0.0, temp_rotor=0.0, timestamp=0.0
            ),
        ]

    def set_commands(self, torques: np.ndarray, pos: np.ndarray, **_: np.ndarray) -> List[SimpleNamespace]:
        self.last_pos = np.array(pos)
        return self.read_states()

    def last_current(self) -> float:
        return self.set_pos_force_limits.call_args.kwargs["current"]


@pytest.fixture
def no_server_thread() -> Iterator[None]:
    with mock.patch.object(motor_chain_robot.threading, "Thread"):
        yield


def make_robot(
    chain: FakeChain,
    hold: float = 30.0,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    limits: np.ndarray = LIMITS,
    **gripper: float,
) -> MotorChainRobot:
    return MotorChainRobot(
        motor_chain=chain,
        use_gravity_comp=False,
        gripper_index=1,
        kp=[10.0, 3.0],
        kd=[1.0, 0.15],
        joint_limits=np.array([[-1.0, 1.0]]),
        gripper_limits=limits,
        gripper_hold_force=hold,
        gripper_type=gripper_type,
        **gripper,
    )


def run(robot: MotorChainRobot, clock: FakeClock, duration: float) -> None:
    for _ in range(round(duration / 0.003)):
        clock.t += 0.003
        robot.update()


def test_robot_switches_gripper_to_pos_force(no_server_thread: None) -> None:
    chain = FakeChain()
    make_robot(chain)
    chain.set_control_mode.assert_called_once_with(1, ControlMode.POS_FORCE)
    kwargs = chain.set_pos_force_limits.call_args.kwargs
    assert kwargs["vel"] == pytest.approx(4.5 * 6.0)
    assert kwargs["current"] == pytest.approx(0.15)
    chain.set_register_float.assert_not_called()


def test_robot_gripper_overrides(no_server_thread: None) -> None:
    chain = FakeChain()
    make_robot(chain, gripper_max_vel=5.0, gripper_max_move_torque=2.0, gripper_kp_pos=73.0)
    chain.set_register_float.assert_called_once_with(1, 27, 73.0)
    kwargs = chain.set_pos_force_limits.call_args.kwargs
    assert kwargs["vel"] == pytest.approx(5.0 * 6.0)
    assert kwargs["current"] == pytest.approx(0.2)


def test_robot_raises_when_mode_rejected(no_server_thread: None) -> None:
    with pytest.raises(RuntimeError, match="rejected"):
        make_robot(FakeChain(reject=True))


def test_robot_clog_switches_current(no_server_thread: None, clock: FakeClock) -> None:
    chain = FakeChain()
    robot = make_robot(chain)
    moving, holding = 0.15, (0.3 + 30.0 * LINEAR_NM_PER_N) / 10.0

    robot.command_joint_pos(np.array([0.0, 0.0]))
    run(robot, clock, 0.03)
    assert chain.last_current() == pytest.approx(moving)
    # the target reaches the motor unmodified
    assert chain.last_pos[1] == 0.0

    chain.gripper_eff = 0.6
    run(robot, clock, 0.06)
    assert chain.last_current() == pytest.approx(holding)
    assert chain.last_pos[1] == 0.0

    robot.command_joint_pos(np.array([0.0, 1.0]))
    run(robot, clock, 0.003)
    assert chain.last_current() == pytest.approx(moving)
    assert chain.last_pos[1] == 6.0


def test_robot_sends_limits_only_on_change(no_server_thread: None, clock: FakeClock) -> None:
    chain = FakeChain()
    robot = make_robot(chain)
    run(robot, clock, 0.03)
    assert chain.set_pos_force_limits.call_count == 1


def test_robot_without_force_limit_uses_full_current(no_server_thread: None, clock: FakeClock) -> None:
    chain = FakeChain()
    robot = make_robot(chain, hold=-1)
    chain.gripper_eff = 0.6
    run(robot, clock, 0.06)
    chain.set_pos_force_limits.assert_called_once()
    assert chain.last_current() == 1.0


def test_robot_crank_hold_current_follows_position(no_server_thread: None, clock: FakeClock) -> None:
    chain = FakeChain()
    chain.gripper_pos = -1.35
    robot = make_robot(chain, gripper_type=GripperType.CRANK_4310, limits=np.array([0.0, -2.7]))
    robot.command_joint_pos(np.array([0.0, 0.0]))
    chain.gripper_eff = 0.6
    run(robot, clock, 0.06)
    assert chain.last_current() == pytest.approx(GripperType.CRANK_4310.force_to_current_pu(30.0, -1.35))

    chain.gripper_pos = -0.1
    # update() limits from the state it read on the previous cycle
    run(robot, clock, 0.006)
    assert chain.last_current() == pytest.approx(GripperType.CRANK_4310.force_to_current_pu(30.0, -0.1))
    assert chain.last_current() < GripperType.CRANK_4310.force_to_current_pu(30.0, -1.35)
