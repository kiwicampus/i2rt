import pytest

from i2rt.flow_base.flow_base_controller import gamepad_teleop_active


def test_no_joystick_runs_on_remote_commands_only() -> None:
    assert gamepad_teleop_active(0, required=False) is False


@pytest.mark.parametrize("required", [False, True])
def test_a_plugged_joystick_enables_teleop(required: bool) -> None:
    assert gamepad_teleop_active(1, required=required) is True


def test_a_required_joystick_that_is_missing_exits() -> None:
    with pytest.raises(SystemExit, match="No joystick/gamepad connected"):
        gamepad_teleop_active(0, required=True)
