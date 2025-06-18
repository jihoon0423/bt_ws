#!/usr/bin/env python3
# bt_runner.py

import os
import time
import rclpy
import py_trees
from py_trees.common import Status, Access
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

import bt_node 
from has_remaining_node import HasRemaining
from count_node import CountNode
from get_waypoint_node import GetWayPoint
from goto_waypoint_node import GotoWaypoint
from capture_node import CaptureNode
from metadata_node import MetadataNode
from return_dock_node import ReturnDock

def wait_for_system_ready(btnode, logger, timeout=30.0):
    logger.info("[Init] System ready 대기 시작")
    start_time = btnode.get_clock().now().nanoseconds * 1e-9
    while rclpy.ok():
        now_time = btnode.get_clock().now().nanoseconds * 1e-9
        if now_time - start_time > timeout:
            logger.warn(f"[Init] 준비 대기 {timeout}s 경과, 강제 시작")
            return True

        # Nav2 액션 서버 준비 확인
        nav2_ready = False
        try:
            client = getattr(btnode, 'nav_to_pose_action_client', None)
            if client is not None and client.server_is_ready():
                nav2_ready = True
        except Exception:
            nav2_ready = False

        # AMCL pose 준비 확인
        amcl_ready = False
        try:
            current = btnode.get_current_pose()
            if isinstance(current, PoseWithCovarianceStamped):
                cov = current.pose.covariance
                if cov[0] < 0.5 and cov[7] < 0.5 and cov[14] < 0.5:
                    amcl_ready = True
        except Exception:
            amcl_ready = False

        # 카메라 이미지 준비 확인
        img_ready = False
        try:
            img = btnode.get_latest_image(timeout_sec=0.0)
            if img is not None:
                img_ready = True
        except Exception:
            img_ready = False

        logger.debug(f"[Init] Nav2_ready={nav2_ready}, AMCL_ready={amcl_ready}, Img_ready={img_ready}")
        if nav2_ready and amcl_ready and img_ready:
            logger.info("[Init] 시스템 준비 완료 (Nav2, AMCL, Camera)")
            return True

        rclpy.spin_once(btnode, timeout_sec=0.1)
    return False

def create_behavior_tree():
    root = py_trees.composites.Sequence("RootSequence", memory=False)

    # LoopSequence: memory=True
    loop_seq = py_trees.composites.Sequence("LoopWaypoint", memory=True)
    has_remain = HasRemaining("HasRemaining")
    get_wp = GetWayPoint("GetWayPoint")
    goto_wp = GotoWaypoint("GotoWaypoint", max_duration=60.0)
    capture = CaptureNode("CaptureNode")
    metadata = MetadataNode("MetadataNode")
    count_after = CountNode("CountNode")
    loop_seq.add_children([has_remain, get_wp, goto_wp, capture, metadata, count_after])

    loop = py_trees.decorators.RepeatUntilFailure(child=loop_seq)

    return_dock = ReturnDock("ReturnDock", max_duration=60.0)

    root.add_children([loop, return_dock])
    return root

def main():
    rclpy.init()
    btnode = bt_node.BTNode()
    logger = btnode.get_logger()

    # Blackboard 설정
    bb = py_trees.blackboard.Client(name="Main")
    bb.register_key(key="bt_node", access=Access.WRITE)
    bb.bt_node = btnode

    bb.register_key(key="initial_pose", access=Access.WRITE)
    bb.register_key(key="waypoints", access=Access.WRITE)
    bb.register_key(key="current_index", access=Access.WRITE)
    bb.register_key(key="photo_paths", access=Access.WRITE)
    bb.photo_paths = []

    bb.register_key(key="current_amcl", access=Access.WRITE)
    bb.current_amcl = None

    bb.register_key(key="goto_failed", access=Access.WRITE)
    bb.goto_failed = False

    # 초기 위치 퍼블리시
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    # TODO: 실제 초기 좌표로 수정
    initial_pose.pose.position.x = -7.731223176932407
    initial_pose.pose.position.y = 0.5009312516255598
    initial_pose.pose.orientation.z = 0.9758351204817772
    initial_pose.pose.orientation.w = 0.21850816377040766

    try:
        btnode.publish_initial_pose(initial_pose)
    except Exception:
        pass
    for _ in range(30):
        rclpy.spin_once(btnode, timeout_sec=0.1)
    bb.initial_pose = initial_pose
    logger.info("[Init] Initial pose published")

    # 웨이포인트 리스트 설정
    waypoints = []
    # 예시 입력: 실제 map 좌표로 수정
    for coords in [
        (-9.29636872541853, 1.1519386275631132, 0.5450941312052887, 0.8383748494113785),
        (-10.343495778706494, 1.8972995638954162, 0.3747301290736029, 0.9271339333475401),
        (-10.497600100859268, 2.8311528221566666, -0.11112884347976208, 0.9938060073006454),
        (-9.645785126844922, 3.232071341419907, -0.8187543454093745, 0.5741439905400618),
        (-9.372491821226403, 2.0881606787739093, 0.8654255401831809, 0.5010375578723113),
    ]:
        wp = PoseStamped()
        wp.header.frame_id = 'map'
        wp.header.stamp = btnode.get_clock().now().to_msg()
        wp.pose.position.x = coords[0]
        wp.pose.position.y = coords[1]
        wp.pose.position.z = 0.0
        wp.pose.orientation.x = 0.0
        wp.pose.orientation.y = 0.0
        wp.pose.orientation.z = coords[2]
        wp.pose.orientation.w = coords[3]
        waypoints.append(wp)
    bb.waypoints = waypoints
    bb.current_index = 0
    logger.info(f"[Init] Waypoints set: {len(waypoints)} points")

    # Nav2 준비 대기
    try:
        btnode.waitUntilNav2Active()
    except Exception:
        pass
    wait_for_system_ready(btnode, logger, timeout=20.0)

    # Behavior Tree 생성 및 setup
    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    logger.info("[Init] Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    # BT tick 루프
    try:
        logger.info("[BT] Initial tick")
        tree.tick()
        while rclpy.ok():
            # AMCL pose 갱신
            try:
                current = btnode.get_current_pose()
                if isinstance(current, PoseWithCovarianceStamped):
                    bb.current_amcl = current
            except Exception:
                pass

            tree.tick()
            # 디버깅: current_index와 total 확인
            idx = getattr(bb, "current_index", None)
            total = len(getattr(bb, "waypoints", []))
            logger.debug(f"[BT Loop] current_index={idx}, total={total}, root.status={root.status}")

            rclpy.spin_once(btnode, timeout_sec=0.1)

            if root.status == Status.SUCCESS:
                logger.info("[BT] All done, exiting.")
                break
    except KeyboardInterrupt:
        logger.info("[BT] Interrupted")
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
