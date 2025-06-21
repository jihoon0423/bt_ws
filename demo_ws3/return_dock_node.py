#!/usr/bin/env python3
# return_dock_node.py

import py_trees
from py_trees.common import Access, Status
import rclpy
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus

class ReturnDock(py_trees.behaviour.Behaviour):
    """
    ReturnDock:
    - initial_pose를 블랙보드에서 읽어 Nav2로 전송
    - 최대 max_duration 초 동안 대기
    - 도착 성공 또는 timeout/실패 시 Status.SUCCESS 반환
    """
    def __init__(self, name="ReturnDock", max_duration=30.0):
        super(ReturnDock, self).__init__(name)
        self.max_duration = max_duration
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.start_time = None

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="initial_pose", access=Access.READ)
        return True

    def initialise(self):
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.start_time = None
        self.logger.info(f"{self.name}: initialise, will return to dock with timeout {self.max_duration}s")

    def update(self):
        bt_node = getattr(self.blackboard, "bt_node", None)
        initial_pose = getattr(self.blackboard, "initial_pose", None)
        if bt_node is None or initial_pose is None:
            self.logger.error(f"{self.name}: bt_node 또는 initial_pose 블랙보드에 없음")
            return Status.SUCCESS

        # 현재 ROS 시간
        try:
            now = bt_node.get_clock().now()
        except Exception:
            try:
                now = rclpy.Clock().now()
            except Exception:
                now = None

        # elapsed 계산 helper
        def get_elapsed():
            if self.start_time is None or now is None:
                return None
            try:
                sec_diff = now.sec - self.start_time.sec
                nsec_diff = now.nanosec - self.start_time.nanosec
                return sec_diff + nsec_diff * 1e-9
            except Exception:
                return None

        # 1) Goal 전송
        if not self.sent:
            # header stamp 갱신
            try:
                initial_pose.header.stamp = bt_node.get_clock().now().to_msg()
            except Exception:
                pass
            send_future = None
            try:
                send_future = bt_node.send_nav_goal(initial_pose)
            except Exception as e:
                self.logger.error(f"{self.name}: send_nav_goal 예외: {e}")
            if send_future is None:
                self.logger.error(f"{self.name}: send_nav_goal 실패, 종료")
                return Status.SUCCESS
            self.sent = True
            self.goal_handle_future = send_future
            if now is not None:
                self.start_time = now
            self.logger.info(f"{self.name}: Return goal sent, waiting acceptance; start_time={self.start_time}")
            return Status.RUNNING

        elapsed = get_elapsed()

        # acceptance 대기
        if self.goal_handle is None:
            if self.goal_handle_future.done():
                try:
                    goal_handle = self.goal_handle_future.result()
                except Exception as e:
                    self.logger.error(f"{self.name}: goal_handle_future.result() 예외: {e}")
                    goal_handle = None
                if goal_handle is None or not getattr(goal_handle, 'accepted', False):
                    self.logger.warning(f"{self.name}: Return goal rejected or invalid")
                    return Status.SUCCESS
                # accepted
                self.goal_handle = goal_handle
                try:
                    self.result_future = goal_handle.get_result_async()
                except Exception as e:
                    self.logger.error(f"{self.name}: get_result_async 예외: {e}")
                    return Status.SUCCESS
                self.logger.info(f"{self.name}: Return goal accepted, waiting result")
                # 즉시 timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: acceptance 후 즉시 timeout, cancel return")
                    try:
                        self.goal_handle.cancel_goal_async()
                    except Exception as e:
                        self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                    return Status.SUCCESS
                return Status.RUNNING
            else:
                # acceptance 기다리는 중: timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: acceptance 대기 중 timeout, skip return")
                    try:
                        if self.goal_handle is not None:
                            self.goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    return Status.SUCCESS
                return Status.RUNNING

        # result 대기
        if self.result_future is not None:
            if self.result_future.done():
                try:
                    status = self.result_future.result().status
                except Exception as e:
                    self.logger.error(f"{self.name}: result_future.result() 예외: {e}")
                    status = None
                if status == GoalStatus.STATUS_SUCCEEDED:
                    self.logger.info(f"{self.name}: Returned to dock successfully")
                else:
                    self.logger.warning(f"{self.name}: Return to dock failed or timed out status={status}")
                return Status.SUCCESS
            else:
                # 아직 결과 기다리는 중: timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: result 대기 중 timeout, cancel return")
                    try:
                        self.goal_handle.cancel_goal_async()
                    except Exception as e:
                        self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                    return Status.SUCCESS
                return Status.RUNNING

        return Status.RUNNING
