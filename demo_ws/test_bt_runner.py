#!/usr/bin/env python3
# bt_runner.py

import os
import re
import cv2
import numpy as np
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

import compare_run_photos   # 모니터링 비교 함수


def create_behavior_tree():
    root = py_trees.composites.Sequence("RootSequence", memory=False)
    fallback = py_trees.composites.Selector("CheckOrProcess", memory=False)
    check_remaining = CheckRemaining("CheckRemaining")
    process_sequence = py_trees.composites.Sequence("ProcessWaypoint", memory=False)
    get_wp = GetWayPoint("GetWayPoint")
    goto_wp = GotoWaypoint("GotoWaypoint")
    capture = CaptureNode("CaptureNode")
    metadata = MetadataNode("MetadataNode")
    count_after = CountNode("CountAfter")
    process_sequence.add_children([get_wp, goto_wp, capture, metadata, count_after])
    fallback.add_children([check_remaining, process_sequence])
    return_dock = ReturnDock("ReturnDock")
    root.add_children([fallback, return_dock])
    return root


def monitor_and_compare(photo_dir, logger):
    # 이전과 동일
    if not os.path.isabs(photo_dir):
        photo_dir = os.path.expanduser(photo_dir)
    if not os.path.isdir(photo_dir):
        try:
            os.makedirs(photo_dir, exist_ok=True)
            logger.info(f"[Monitor] Created directory: {photo_dir}")
        except Exception as e:
            logger.error(f"[Monitor] Failed to create photo_dir {photo_dir}: {e}")
            return
    try:
        files = os.listdir(photo_dir)
    except Exception as e:
        logger.error(f"[Monitor] Failed to list directory {photo_dir}: {e}")
        return
    logger.debug(f"[Monitor] Current files in {photo_dir}: {files}")
    for fname in files:
        m = re.match(r'waypoint_(\d+)\.png$', fname)
        if not m:
            continue
        idx = m.group(1)
        waypoint_path = os.path.join(photo_dir, fname)
        ref_filename = f"reference{idx}.png"
        ref_path = os.path.join(photo_dir, ref_filename)
        if not os.path.isfile(ref_path):
            logger.warning(f"[Monitor] reference not found for idx={idx}: {ref_path}")
            continue
        out_filename = f"comparison_result_{idx}.png"
        out_path = os.path.join(photo_dir, out_filename)
        if os.path.isfile(out_path):
            logger.debug(f"[Monitor] comparison_result for idx={idx} already exists; skip")
            continue
        ref_img = cv2.imread(ref_path)
        target_img = cv2.imread(waypoint_path)
        if ref_img is None or target_img is None:
            logger.error(f"[Monitor] Failed to load images for idx={idx}")
            continue
        logger.info(f"[Monitor] Comparing reference{idx}.png <-> waypoint_{idx}.png ...")
        try:
            result = compare_run_photos.detect_changes(ref_img, target_img)
        except Exception as e:
            logger.error(f"[Monitor] detect_changes error for idx={idx}: {e}")
            continue
        try:
            ok = cv2.imwrite(out_path, result)
            if ok:
                logger.info(f"[Monitor] Saved comparison_result_{idx}.png")
            else:
                logger.error(f"[Monitor] Failed to save comparison image for idx={idx}")
        except Exception as e:
            logger.error(f"[Monitor] Error saving comparison image for idx={idx}: {e}")


def display_dashboard(photo_dir, logger):
    # 이전과 동일 (생략)
    pass  # 필요시 앞 예시 코드 참고


def main():
    rclpy.init()
    btnode = bt_node.BTNode()
    logger = btnode.get_logger()

    # Blackboard Client 설정
    bb_client = py_trees.blackboard.Client(name="Main")
    bb_client.register_key(key="bt_node", access=Access.WRITE)
    bb_client.bt_node = btnode
    bb_client.register_key(key="initial_pose", access=Access.WRITE)
    bb_client.register_key(key="waypoints", access=Access.WRITE)
    bb_client.register_key(key="current_index", access=Access.WRITE)
    bb_client.register_key(key="photo_paths", access=Access.WRITE)
    bb_client.photo_paths = []
    # current_amcl 키 등록 및 초기화
    bb_client.register_key(key="current_amcl", access=Access.WRITE)
    bb_client.current_amcl = None

    # 1) AMCL 초기 위치 설정
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    # 하드코딩 좌표 예시
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
    # AMCL 수렴 대기
    for _ in range(20):
        rclpy.spin_once(btnode, timeout_sec=0.1)
    bb_client.initial_pose = initial_pose
    logger.info("Initial pose set and stored on blackboard")

    # 2) waypoints 설정
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
    # 추가 waypoint...
    bb_client.waypoints = waypoints
    bb_client.current_index = 0
    logger.info(f"Waypoints set on blackboard: {len(waypoints)} points")

    # 3) Behavior Tree 생성 및 setup
    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    logger.info("Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    # photo_dir: CaptureNode/MetadataNode/모니터링이 사용하는 동일 디렉토리
    photo_dir = '/home/choi/waypoint_photos'

    try:
        logger.info("Initial tick of BehaviorTree")
        tree.tick()

        # monitor_and_compare(photo_dir, logger)  # 필요 시 호출

        while rclpy.ok():
            # AMCL pose를 블랙보드에 갱신: 반드시 current_amcl이 None이 아닌 실제 PoseWithCovarianceStamped 또는 Pose 객체로 설정되어야 함
            try:
                current = btnode.get_current_pose()
                if isinstance(current, PoseWithCovarianceStamped) or isinstance(current, Pose):
                    bb_client.current_amcl = current
                else:
                    # None 또는 다른 타입일 경우엔 건너뜀
                    pass
            except Exception:
                pass

            tree.tick()
            # monitor_and_compare(photo_dir, logger)  # 필요 시 호출

            rclpy.spin_once(btnode, timeout_sec=0.1)
            if root.status == Status.SUCCESS:
                logger.info("Behavior Tree 전체 완료, 종료합니다.")
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
