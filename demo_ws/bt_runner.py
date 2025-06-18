#!/usr/bin/env python3
# bt_runner.py

import os
import rclpy
import py_trees
from py_trees.common import Status, Access
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose

import bt_node 
from check_remaining_node import CheckRemaining
from count_node import CountNode
from get_waypoint_node import GetWayPoint
from goto_waypoint_node import GotoWaypoint
from capture_node import CaptureNode
from metadata_node import MetadataNode
from return_dock_node import ReturnDock


def create_behavior_tree():
    """
    Behavior Tree 구조:
    RootSequence
      ├── Selector(CheckOrProcess)
      │     ├── CheckRemaining  (남은 waypoint 없으면 SUCCESS)
      │     └── Sequence:
      │           GetWayPoint -> GotoWaypoint(timeout) -> CaptureNode -> MetadataNode -> CountNode
      ├── ReturnDock(timeout)
    """
    root = py_trees.composites.Sequence("RootSequence", memory=False)

    fallback = py_trees.composites.Selector("CheckOrProcess", memory=False)
    check_remaining = CheckRemaining("CheckRemaining")

    process_sequence = py_trees.composites.Sequence("ProcessWaypoint", memory=False)
    get_wp = GetWayPoint("GetWayPoint")
    # GotoWaypoint: max_duration=30초
    goto_wp = GotoWaypoint("GotoWaypoint", max_duration=30.0)
    capture = CaptureNode("CaptureNode")
    metadata = MetadataNode("MetadataNode")
    count_after = CountNode("CountNode")
    process_sequence.add_children([get_wp, goto_wp, capture, metadata, count_after])

    fallback.add_children([check_remaining, process_sequence])

    # ReturnDock: initial_pose로 돌아가기, timeout 설정 (예: 30초)
    return_dock = ReturnDock("ReturnDock", max_duration=30.0)

    root.add_children([fallback, return_dock])
    return root


def main():
    rclpy.init()
    btnode = bt_node.BTNode()
    logger = btnode.get_logger()

    bb_client = py_trees.blackboard.Client(name="Main")
    bb_client.register_key(key="bt_node", access=Access.WRITE)
    bb_client.bt_node = btnode

    bb_client.register_key(key="initial_pose", access=Access.WRITE)
    bb_client.register_key(key="waypoints", access=Access.WRITE)
    bb_client.register_key(key="current_index", access=Access.WRITE)
    bb_client.register_key(key="photo_paths", access=Access.WRITE)
    bb_client.photo_paths = []

    bb_client.register_key(key="current_amcl", access=Access.WRITE)
    bb_client.current_amcl = None

    bb_client.register_key(key="goto_failed", access=Access.WRITE)
    bb_client.goto_failed = False

    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    initial_pose.pose.position.x = -0.3289598865580746
    initial_pose.pose.position.y = -0.09526620349955373
    initial_pose.pose.position.z = 0.0
    initial_pose.pose.orientation.x = 0.0
    initial_pose.pose.orientation.y = 0.0
    initial_pose.pose.orientation.z = -0.9997488323672816
    initial_pose.pose.orientation.w = 0.022411429679006712

    try:
        btnode.publish_initial_pose(initial_pose)
    except Exception:
        pass
    for _ in range(20):
        rclpy.spin_once(btnode, timeout_sec=0.1)
    bb_client.initial_pose = initial_pose
    logger.info("Initial pose published and stored on blackboard")

    waypoints = []
    wp1 = PoseStamped()
    wp1.header.frame_id = 'map'
    wp1.header.stamp = btnode.get_clock().now().to_msg()
    wp1.pose.position.x = -1.9927422064708293
    wp1.pose.position.y = 3.6205491946654473
    wp1.pose.position.z = 0.0
    wp1.pose.orientation.x = 0.0
    wp1.pose.orientation.y = 0.0
    wp1.pose.orientation.z = 0.5438154560388769
    wp1.pose.orientation.w = 0.8392048318338189
    waypoints.append(wp1)

    wp2 = PoseStamped()
    wp2.header.frame_id = 'map'
    wp2.header.stamp = btnode.get_clock().now().to_msg()
    wp2.pose.position.x = 2.0
    wp2.pose.position.y = 2.0
    wp2.pose.position.z = 0.0
    wp2.pose.orientation.x = 0.0
    wp2.pose.orientation.y = 0.0
    wp2.pose.orientation.z = 0.0
    wp2.pose.orientation.w = 1.0
    waypoints.append(wp2)

    wp3 = PoseStamped()
    wp3.header.frame_id = 'map'
    wp3.header.stamp = btnode.get_clock().now().to_msg()
    wp3.pose.position.x = 5.0
    wp3.pose.position.y = -1.0
    wp3.pose.position.z = 0.0
    wp3.pose.orientation.x = 0.0
    wp3.pose.orientation.y = 0.0
    wp3.pose.orientation.z = 0.0
    wp3.pose.orientation.w = 1.0
    waypoints.append(wp3)

    bb_client.waypoints = waypoints
    bb_client.current_index = 0
    logger.info(f"Waypoints set on blackboard: {len(waypoints)} points")

    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    logger.info("Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    photo_dir = os.path.expanduser('~/waypoint_project/waypoint_photos')

    try:
        logger.info("Initial tick of BehaviorTree")
        tree.tick()

        while rclpy.ok():
            # AMCL pose를 블랙보드에 갱신
            try:
                current = btnode.get_current_pose()
                if isinstance(current, PoseWithCovarianceStamped) or isinstance(current, Pose):
                    bb_client.current_amcl = current
            except Exception:
                pass

            tree.tick()
            rclpy.spin_once(btnode, timeout_sec=0.1)

            if root.status == Status.SUCCESS:
                logger.info("Behavior Tree 전체 완료, 종료합니다.")
                # (필요 시) 비교 결과 대시보드 표시 함수 호출
                # display_dashboard(photo_dir, logger)
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: 중단")
    finally:
        try:
            if hasattr(btnode, 'lifecycleShutdown'):
                btnode.lifecycleShutdown()
            btnode.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
