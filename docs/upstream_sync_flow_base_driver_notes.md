# Flow Base driver migration notes: upstream sync v1.3.6

This fork was synced with `i2rt-robotics/i2rt` main (v1.3.6). Two behavior changes in
`i2rt/flow_base/flow_base_controller.py` affect any client that reads odometry or sends
velocity commands, in particular the `rtop` ROS 2 driver (`base_handler.py` and
`flow_base_utils.py`). The fork now carries the upstream implementation for both.

## 1. Body frame sign convention (AXIS_SIGN)

### Before the sync

The kinematic model inherited from tidybot2 had the caster layout mirrored about the x axis
relative to the physical base. As a result the FlowBase body frame was:

| Axis | Positive direction |
|------|--------------------|
| x | forward |
| y | right |
| theta | clockwise |

The driver compensated on its side with `rep103_signs: [1.0, -1.0, -1.0, 1.0]` on outgoing
commands, and by negating `y`, `theta`, `vy` and `vtheta` when parsing odometry.

### After the sync

Upstream rewrote the caster positions `h_x, h_y` to the true physical layout and introduced
`AXIS_SIGN`, a per axis sign vector applied symmetrically to the odometry output and the command
input. It is now the identity `[1.0, 1.0, 1.0]`, kept as an explicit hook. The FlowBase body
frame now follows REP 103 directly:

| Axis | Positive direction |
|------|--------------------|
| x | forward |
| y | left |
| theta | counterclockwise |

Odometry still equals the command, so a `set_target_velocity([0, 0.1, 0], frame="local")`
moves the base to its left and odometry reports positive `vy`.

### Driver action

Remove the sign compensation, otherwise y and yaw are flipped twice:

- `rep103_signs` becomes `[1.0, 1.0, 1.0, 1.0]` (or the option is dropped).
- The odometry pose and twist parsers stop negating `y`, `theta`, `vy` and `vtheta`.
- The odom TF broadcast test fixture must use the new schema below.

## 2. get_odometry() return schema

### Before the sync (fork)

```python
{
    "translation": np.ndarray,       # shape (2,)  [x, y] in the world (odom) frame, meters
    "rotation": float,               # theta in the world frame, radians
    "linear_velocity": np.ndarray,   # shape (2,)  [vx, vy] in the body frame, m/s
    "angular_velocity": float,       # vtheta in the body frame, rad/s
}
```

### After the sync (upstream)

```python
{
    "position": {
        "translation": np.ndarray,   # shape (3,)  [x, y, 0.0] in the world (odom) frame, meters
        "rotation": float,           # theta in the world frame, radians
    },
    "velocity": {
        "world": {
            "translation": np.ndarray,   # shape (3,)  [vx, vy, 0.0] in the world frame, m/s
            "rotation": float,           # vtheta, rad/s
        },
        "body": {
            "translation": np.ndarray,   # shape (3,)  [vx, vy, 0.0] in the body frame, m/s
            "rotation": float,           # vtheta, rad/s
        },
    },
}
```

Key mapping from the old flat schema:

| Old key | New key |
|---------|---------|
| `translation` (2D) | `position.translation` (3D, take `[:2]`) |
| `rotation` | `position.rotation` |
| `linear_velocity` (2D, body) | `velocity.body.translation` (3D, take `[:2]`) |
| `angular_velocity` (body) | `velocity.body.rotation` |
| not available | `velocity.world.translation`, `velocity.world.rotation` |

Notes:

- Vectors are 3D with a zero z component. The values arrive over the `portal` RPC as numpy
  arrays or lists depending on the client, so index them rather than unpacking.
- The body frame twist is what a `TwistStamped` with `child_frame_id == base_link` expects, so
  the driver should keep publishing `velocity.body`. The world frame twist is new and optional.
- Odometry is now integrated over the measured loop period instead of the nominal 5 ms, so pose
  no longer under integrates when the control loop runs slow. No driver change is needed for this.

### Driver action

- `parse_odometry_pose`: read `odo["position"]["translation"][0:2]` and `odo["position"]["rotation"]`.
- `parse_odometry_twist`: read `odo["velocity"]["body"]["translation"][0:2]` and
  `odo["velocity"]["body"]["rotation"]`.
- The zero filled fallback odometry dict in `flow_base_utils.py` must mirror the new nested shape.
- Update the mocked return value in `test_odom_tf_broadcast.py` to the new nested shape.

## 3. Related changes worth knowing about

- `LinearRailVehicle.__init__` now accepts `control_freq` (default 200 Hz), `check_caster_steering`
  and `usb_gpio_device`. Existing keyword arguments keep their meaning.
- A new `get_wheel_states()` RPC exposes per motor position, velocity and effort for the 8 base
  motors, split into `steer` and `drive`.
- A caster steering fault ramps the base to a stop and makes the controller process exit with
  code 2. Clients can poll `caster_fault()`.
- The linear rail GPIO backend detects a Raspberry Pi through the device tree model string, so
  a Jetson or other ARM board no longer tries to import `RPi.GPIO`. It uses the USB to GPIO
  converter backend instead, selected by `usb_gpio_device` or the `I2RT_USB_GPIO_PORT` variable.
