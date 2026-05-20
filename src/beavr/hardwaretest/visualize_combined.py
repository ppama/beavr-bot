#!/usr/bin/env python3
"""
Combined PyBullet visualizer for the XArm7 + LEAP hand teleoperation.

Subscribes to:
  - XArm7 joint state  : port 10010 (XARM_ENDEFF_PUBLISH_PORT), topic "xarm7_right"
                         payload: plain dict with key "joint_angles_rad" -> list[7 floats]
  - LEAP hand joints   : port 8120  (CARTESIAN_COMMAND_PUBLISHER_PORT), topic "joint_angles"
                         payload: JointTarget with joint_positions_rad -> list[16 floats]

Renders both in real time using the combined leap_xarm7.urdf (arm + hand as a single body).

Run alongside the main teleop stack (with or without hardware):
    python src/beavr/hardwaretest/visualize_combined.py [--host HOST]
"""

import argparse
import os
import sys
import time

import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.components.operator.operator_types import JointTarget
from beavr.teleop.configs.constants.ports import (
    CARTESIAN_COMMAND_PUBLISHER_PORT,
    XARM_ENDEFF_PUBLISH_PORT,
)

# ── XArm7 joint indices in the combined URDF ─────────────────────────────────
# joint1..joint7 are PyBullet joints 1-7 (same as DOF order, no fixed joints before them
# except the world_joint at 0).
_XARM_JOINTS = [1, 2, 3, 4, 5, 6, 7]

# ── LEAP hand joint indices in the combined URDF ─────────────────────────────
# After world_joint(0) + joint1-7(1-7) + joint_eef(8) + hand_mount(9) + base_joint(10),
# the LEAP revolute joints appear at:
#   index chain:  a(11) b(12) c(13) d(14)   [index_tip fixed at 15]
#   middle chain: e(16) f(17) g(18) h(19)   [middle_tip fixed at 20]
#   ring chain:   i(21) j(22) k(23) l(24)   [ring_tip fixed at 25]
#   thumb chain:  n(26) m(27) o(28) p(29)   [thumb_tip fixed at 30]
#
# Motor index → PyBullet joint index mapping:
#   Index:  no swap         (motors 0-3  → joints a,b,c,d = 11,12,13,14)
#   Middle: side/fwd swap   (motors 4-7  → joints f,e,g,h = 17,16,18,19)
#   Ring:   side/fwd swap   (motors 8-11 → joints j,i,k,l = 22,21,23,24)
#   Thumb:  no swap         (motors 12-15→ joints n,m,o,p = 26,27,28,29)
_LEAP_MOTOR_TO_JOINT = [
    11,
    12,
    13,
    14,  # index  (no swap)
    17,
    16,
    18,
    19,  # middle (swapped: motor4→f, motor5→e)
    22,
    21,
    23,
    24,  # ring   (swapped: motor8→j, motor9→i)
    26,
    27,
    28,
    29,  # thumb  (no swap)
]


def _build_robot(urdf_path: str) -> int:
    body = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    for joint_idx in _XARM_JOINTS + _LEAP_MOTOR_TO_JOINT:
        p.resetJointState(body, joint_idx, 0.0)
    return body


def _apply_xarm_angles(body: int, angles) -> None:
    angles = list(angles)
    for i, joint_idx in enumerate(_XARM_JOINTS):
        if i < len(angles):
            p.resetJointState(body, joint_idx, float(angles[i]))


def _apply_leap_motors(body: int, motor_angles) -> None:
    motor_angles = list(motor_angles)
    for motor_idx, joint_idx in enumerate(_LEAP_MOTOR_TO_JOINT):
        if motor_idx < len(motor_angles):
            p.resetJointState(body, joint_idx, float(motor_angles[motor_idx]))


def _extract_xarm_angles(msg):
    """Pull 7 joint angles from the xarm state dict (plain Python dict)."""
    if msg is None:
        return None
    if isinstance(msg, dict):
        # Prefer the convenience key published directly
        if "joint_angles_rad" in msg and msg["joint_angles_rad"] is not None:
            return msg["joint_angles_rad"]
        # Fall back to the joint_states sub-dict
        js = msg.get("joint_states")
        if isinstance(js, dict) and js.get("joint_position") is not None:
            return js["joint_position"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Combined XArm7 + LEAP hand PyBullet visualizer")
    parser.add_argument(
        "--host", default="localhost", help="Host running the beavr teleop stack (default: localhost)"
    )
    parser.add_argument(
        "--xarm-port",
        type=int,
        default=XARM_ENDEFF_PUBLISH_PORT,
        help=f"ZMQ port for xarm state (default: {XARM_ENDEFF_PUBLISH_PORT})",
    )
    parser.add_argument(
        "--xarm-topic", default="xarm7_right", help='ZMQ topic for xarm state (default: "xarm7_right")'
    )
    parser.add_argument(
        "--leap-port",
        type=int,
        default=CARTESIAN_COMMAND_PUBLISHER_PORT,
        help=f"ZMQ port for LEAP hand joints (default: {CARTESIAN_COMMAND_PUBLISHER_PORT})",
    )
    args = parser.parse_args()

    # ── PyBullet GUI ──────────────────────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # Camera: slightly above and to the side so the full arm+hand is visible
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0.3, 0, 0.4],
    )

    # Optional floor plane for orientation reference
    p.loadURDF("plane.urdf", [0, 0, -0.01])

    # ── Load combined URDF ────────────────────────────────────────────────────
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../../../assets/urdf/leap_xarm7/leap_xarm7.urdf",
    )
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}", file=sys.stderr)
        sys.exit(1)

    robot = _build_robot(urdf_path)

    # ── ZMQ subscribers ───────────────────────────────────────────────────────
    xarm_sub = ZMQSubscriber(
        host=args.host,
        port=args.xarm_port,
        topic=args.xarm_topic,
        # No message_type: xarm publishes a plain dict
    )
    leap_sub = ZMQSubscriber(
        host=args.host,
        port=args.leap_port,
        topic="joint_angles",
        message_type=JointTarget,
    )

    print(f"XArm7  : tcp://{args.host}:{args.xarm_port}  topic='{args.xarm_topic}'")
    print(f"LEAP   : tcp://{args.host}:{args.leap_port}  topic='joint_angles'")
    print("Move your hand in VR — both robot and hand will follow.")
    print("Close the PyBullet window or press Ctrl-C to exit.\n")

    last_xarm = None
    last_leap = None

    try:
        while p.isConnected():
            # ── Receive latest messages (non-blocking) ────────────────────────
            xarm_msg = xarm_sub.recv_keypoints()
            leap_msg = leap_sub.recv_keypoints()

            # Cache last valid values so the display doesn't freeze when one
            # stream momentarily lags.
            xarm_angles = _extract_xarm_angles(xarm_msg)
            if xarm_angles is not None:
                last_xarm = xarm_angles

            if leap_msg is not None:
                if isinstance(leap_msg, JointTarget):
                    last_leap = leap_msg.joint_positions_rad
                elif isinstance(leap_msg, dict) and "joint_positions_rad" in leap_msg:
                    last_leap = leap_msg["joint_positions_rad"]
                else:
                    try:
                        last_leap = list(leap_msg)
                    except TypeError:
                        pass

            # ── Apply angles ──────────────────────────────────────────────────
            if last_xarm is not None:
                _apply_xarm_angles(robot, last_xarm)
            if last_leap is not None:
                _apply_leap_motors(robot, last_leap)

            p.stepSimulation()
            time.sleep(0.005)  # ~200 Hz render cap

    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()
        xarm_sub.stop()
        leap_sub.stop()


if __name__ == "__main__":
    main()
