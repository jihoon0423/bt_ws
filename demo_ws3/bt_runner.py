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
      │           GetWayPoint -> GotoWaypoint(timeout=60s) -> CaptureNode -> MetadataNode -> CountNode
      ├── ReturnDock(timeout=60s)  # 초기 위치로 복귀, 타임아웃 60초
    """
    root = py_trees.composites.Sequence("RootSequence", memory=False)

    # 1) 남은 웨이포인트 체크 또는 처리 시퀀스
    fallback = py_trees.composites.Selector("CheckOrProcess", memory=False)
    check_remaining = CheckRemaining("CheckRemaining")

    process_sequence = py_trees.composites.Sequence("ProcessWaypoint", memory=False)
    get_wp = GetWayPoint("GetWayPoint")
    # GotoWaypoint: max_duration=60초, wait_duration는 GotoWaypoint 내부 기본(예: 2초) 사용
    goto_wp = GotoWaypoint("GotoWaypoint", max_duration=60.0)
    capture = CaptureNode("CaptureNode")
    metadata = MetadataNode("MetadataNode")
    count_after = CountNode("CountNode")
    process_sequence.add_children([get_wp, goto_wp, capture, metadata, count_after])

    fallback.add_children([check_remaining, process_sequence])

    # 2) ReturnDock: 초기 위치로 복귀, timeout 60초 지정
    return_dock = ReturnDock("ReturnDock", max_duration=60.0)

    root.add_children([fallback, return_dock])
    return root

def main():
    rclpy.init()
    btnode = bt_node.BTNode()
    logger = btnode.get_logger()

    # Blackboard 메인 클라이언트 (쓰기 권한)
    bb_client = py_trees.blackboard.Client(name="Main")
    bb_client.register_key(key="bt_node", access=Access.WRITE)
    bb_client.bt_node = btnode

    bb_client.register_key(key="initial_pose", access=Access.WRITE)
    bb_client.register_key(key="waypoints", access=Access.WRITE)
    bb_client.register_key(key="current_index", access=Access.WRITE)
    bb_client.register_key(key="photo_paths", access=Access.WRITE)
    bb_client.photo_paths = []

    # AMCL pose를 기록할 키
    bb_client.register_key(key="current_amcl", access=Access.WRITE)
    bb_client.current_amcl = None

    # GotoWaypoint 실패 플래그
    bb_client.register_key(key="goto_failed", access=Access.WRITE)
    bb_client.goto_failed = False

    # 1) 초기 위치 설정 (예시: map 프레임 좌표)
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    # TODO: 실제 원하는 초기 좌표로 수정
    initial_pose.pose.position.x = -7.731223176932407
    initial_pose.pose.position.y = 0.5009312516255598
    initial_pose.pose.position.z = 0.0
    initial_pose.pose.orientation.x = 0.0
    initial_pose.pose.orientation.y = 0.0
    initial_pose.pose.orientation.z = 0.9758351204817772
    initial_pose.pose.orientation.w = 0.21850816377040766

    # AMCL 초기 퍼블리시
    try:
        btnode.publish_initial_pose(initial_pose)
    except Exception:
        pass
    # 퍼블리시 후 잠시 spin하여 AMCL 반영
    for _ in range(20):
        rclpy.spin_once(btnode, timeout_sec=0.1)
    bb_client.initial_pose = initial_pose
    logger.info("Initial pose published and stored on blackboard")

    # 2) 웨이포인트 리스트 설정
    waypoints = []
    # 예시로 5개 좌표 추가; 실제 좌표 값은 echo한 AMCL 값 혹은 테스트한 map 좌표 사용
    wp1 = PoseStamped()
    wp1.header.frame_id = 'map'
    wp1.header.stamp = btnode.get_clock().now().to_msg()
    wp1.pose.position.x = -9.29636872541853
    wp1.pose.position.y = 1.1519386275631132
    wp1.pose.position.z = 0.0
    wp1.pose.orientation.x = 0.0
    wp1.pose.orientation.y = 0.0
    wp1.pose.orientation.z = 0.5450941312052887
    wp1.pose.orientation.w = 0.8383748494113785
    waypoints.append(wp1)

    wp2 = PoseStamped()
    wp2.header.frame_id = 'map'
    wp2.header.stamp = btnode.get_clock().now().to_msg()
    wp2.pose.position.x = -10.343495778706494
    wp2.pose.position.y = 1.8972995638954162
    wp2.pose.position.z = 0.0
    wp2.pose.orientation.x = 0.0
    wp2.pose.orientation.y = 0.0
    wp2.pose.orientation.z = 0.3747301290736029
    wp2.pose.orientation.w = 0.9271339333475401
    waypoints.append(wp2)

    wp3 = PoseStamped()
    wp3.header.frame_id = 'map'
    wp3.header.stamp = btnode.get_clock().now().to_msg()
    wp3.pose.position.x = -10.497600100859268
    wp3.pose.position.y = 2.8311528221566666
    wp3.pose.position.z = 0.0
    wp3.pose.orientation.x = 0.0
    wp3.pose.orientation.y = 0.0
    wp3.pose.orientation.z = -0.11112884347976208
    wp3.pose.orientation.w = 0.9938060073006454
    waypoints.append(wp3)

    wp4 = PoseStamped()
    wp4.header.frame_id = 'map'
    wp4.header.stamp = btnode.get_clock().now().to_msg()
    wp4.pose.position.x = -9.645785126844922
    wp4.pose.position.y = 3.232071341419907
    wp4.pose.position.z = 0.0
    wp4.pose.orientation.x = 0.0
    wp4.pose.orientation.y = 0.0
    wp4.pose.orientation.z = -0.8187543454093745
    wp4.pose.orientation.w = 0.5741439905400618
    waypoints.append(wp4)

    wp5 = PoseStamped()
    wp5.header.frame_id = 'map'
    wp5.header.stamp = btnode.get_clock().now().to_msg()
    wp5.pose.position.x = -9.372491821226403
    wp5.pose.position.y = 2.0881606787739093
    wp5.pose.position.z = 0.0
    wp5.pose.orientation.x = 0.0
    wp5.pose.orientation.y = 0.0
    wp5.pose.orientation.z = 0.8654255401831809
    wp5.pose.orientation.w = 0.5010375578723113
    waypoints.append(wp5)

    bb_client.waypoints = waypoints
    bb_client.current_index = 0
    logger.info(f"Waypoints set on blackboard: {len(waypoints)} points")

    # 3) Behavior Tree 생성 및 setup
    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    logger.info("Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    # 4) BT tick & AMCL 갱신 루프
    try:
        logger.info("Initial tick of BehaviorTree")
        tree.tick()

        while rclpy.ok():
            # AMCL pose를 주기적으로 블랙보드에 저장
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
                # 필요 시: 비교결과나 대시보드 표시 호출
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
