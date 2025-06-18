#!/usr/bin/env python3
# bt_runner.py

import os
import rclpy
import py_trees
from py_trees.common import Status, Access
from geometry_msgs.msg import PoseStamped

import bt_node  
from check_remaining_node import CheckRemaining
from count_node import CountNode
from get_waypoint_node import GetWayPoint
from goto_waypoint_node import GotoWaypoint
from capture_node import CaptureNode
from return_dock_node import ReturnDock
from compare_node import CompareNode
from dashboard_node import DashboardNode  

def create_behavior_tree():
    """
    Behavior Tree 구조:
    RootSequence
      ├── Selector(CheckOrProcess)
      │     ├── CheckRemaining (남은 waypoint 없으면 SUCCESS)
      │     └── Sequence(GetWayPoint -> GotoWaypoint -> CaptureNode -> CountNode)
      ├── ReturnDock
      ├── CompareNode
      └── DashboardNode
    """
    root = py_trees.composites.Sequence("RootSequence", memory=False)


    fallback = py_trees.composites.Selector("CheckOrProcess", memory=False)
    check_remaining = CheckRemaining("CheckRemaining")
    process_sequence = py_trees.composites.Sequence("ProcessWaypoint", memory=False)
    get_wp = GetWayPoint("GetWayPoint")
    goto_wp = GotoWaypoint("GotoWaypoint")
    capture = CaptureNode("CaptureNode")
    count_after = CountNode("CountAfter")
    process_sequence.add_children([get_wp, goto_wp, capture, count_after])
    fallback.add_children([check_remaining, process_sequence])


    return_dock = ReturnDock("ReturnDock")


    compare_node = CompareNode("CompareNode")

    dashboard_node = DashboardNode("DashboardNode")


    root.add_children([fallback, return_dock, compare_node, dashboard_node])
    return root

def main():
    rclpy.init()
    btnode = bt_node.BTNode()


    bb_client = py_trees.blackboard.Client(name="Main")
    bb_client.register_key(key="bt_node", access=Access.WRITE)
    bb_client.bt_node = btnode
    bb_client.register_key(key="initial_pose", access=Access.WRITE)
    bb_client.register_key(key="waypoints", access=Access.WRITE)
    bb_client.register_key(key="current_index", access=Access.WRITE)
    bb_client.register_key(key="photo_paths", access=Access.WRITE)
    bb_client.photo_paths = []  

   
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    initial_pose.pose.position.x = -0.3527659295151565
    initial_pose.pose.position.y = -3.292957691502578e-11
    initial_pose.pose.position.z = 0.0
    initial_pose.pose.orientation.x = 0.0
    initial_pose.pose.orientation.y = 0.0
    initial_pose.pose.orientation.z = 0.9981621357977354
    initial_pose.pose.orientation.w = -0.06059992293479631

    btnode.publish_initial_pose(initial_pose)
    try:
        btnode.waitUntilNav2Active()
    except AttributeError:
        pass
    bb_client.initial_pose = initial_pose
    btnode.get_logger().info("Initial pose set and stored on blackboard")


    waypoints = []

    wp1 = PoseStamped()
    wp1.header.frame_id = 'map'
    wp1.header.stamp = btnode.get_clock().now().to_msg()
    wp1.pose.position.x = 4.574370434309949
    wp1.pose.position.y = 5.194985176338236
    wp1.pose.position.z = 0.0
    wp1.pose.orientation.x = 0.0
    wp1.pose.orientation.y = 0.0
    wp1.pose.orientation.z = 0.1533114364715118
    wp1.pose.orientation.w = 0.9881779209469526
    waypoints.append(wp1)

    wp2 = PoseStamped()
    wp2.header.frame_id = 'map'
    wp2.header.stamp = btnode.get_clock().now().to_msg()
    wp2.pose.position.x = 2.8336787738753957
    wp2.pose.position.y = -1.7789931122781792
    wp2.pose.position.z = 0.0
    wp2.pose.orientation.x = 0.0
    wp2.pose.orientation.y = 0.0
    wp2.pose.orientation.z = -0.8518522017756794
    wp2.pose.orientation.w = 0.5237822317814219
    waypoints.append(wp2)



    bb_client.waypoints = waypoints
    bb_client.current_index = 0
    btnode.get_logger().info(f"Waypoints set on blackboard: {len(waypoints)} points")

    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.INFO
    btnode.get_logger().info("Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    try:
        btnode.get_logger().info("Initial tick of BehaviorTree")
        tree.tick()
        while rclpy.ok():
            tree.tick()
            rclpy.spin_once(btnode, timeout_sec=0.1)
            if root.status == Status.SUCCESS:
                btnode.get_logger().info("Behavior Tree 전체 완료, 종료합니다.")
                break
    except KeyboardInterrupt:
        btnode.get_logger().info("KeyboardInterrupt: 중단")
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
