#!/usr/bin/env python3
# goto_waypoint_node.py

import py_trees
from py_trees.common import Access, Status
import rclpy
from action_msgs.msg import GoalStatus
import time

class GotoWaypoint(py_trees.behaviour.Behaviour):
    """
    GotoWaypoint:
    - current_waypoint로 Nav2 goal 전송
    - goal 도달 성공 시, 도착 인지 순간부터 wait_duration(예:2초)간 대기 후 SUCCESS 반환
    - goal accepted 후 max_duration(예:60초) 이내에 결과가 오지 않으면 cancel & 실패 처리:
        * blackboard.goto_failed = True 로 표시
        * Status.SUCCESS 혹은 FAILURE 반환 (여기서는 다음 CaptureNode로 넘어가기 위해 SUCCESS 반환)
    - goal rejected 또는 즉시 실패 시: blackboard.goto_failed = True, Status.SUCCESS 반환
    """
    def __init__(self, name="GotoWaypoint", max_duration=60.0, wait_duration=2.0):
        """
        :param name: 노드 이름
        :param max_duration: goal 전송 후 도착(success) 또는 실패(failure) 판정까지 최대 대기 시간(초)
        :param wait_duration: 도착 성공 시 머무를 시간(초)
        """
        super(GotoWaypoint, self).__init__(name)
        # 상태 변수
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.arrival_time = None
        self.accepted = False
        # 파라미터
        self.max_duration = max_duration
        self.wait_duration = wait_duration
        # 시간 기록용
        self.start_time = None

    def setup(self, timeout=None):
        """
        블랙보드 클라이언트 생성 및 필요한 키 등록
        """
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        # Nav2 제어용 BTNode
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        # current_waypoint PoseStamped
        self.blackboard.register_key(key="current_waypoint", access=Access.READ)
        # 실패 플래그 기록용
        self.blackboard.register_key(key="goto_failed", access=Access.WRITE)
        return True

    def initialise(self):
        """
        매 tick sequence 진입 시 초기화
        """
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.arrival_time = None
        self.accepted = False
        self.start_time = None
        # 실패 플래그 초기 리셋
        try:
            self.blackboard.goto_failed = False
        except Exception:
            pass
        self.logger.info(f"{self.name}: initialise, max_duration={self.max_duration}s, wait_duration={self.wait_duration}s")

    def update(self):
        """
        1) Goal 전송
        2) Acceptance 대기
        3) Result 대기: 성공 시 arrival_time 기록 후 wait_duration 동안 RUNNING -> SUCCESS
                        실패 시 goto_failed=True, SUCCESS 반환 (다음 CaptureNode로)
        4) Timeout(max_duration) 초과 시: cancel goal, goto_failed=True, SUCCESS 반환
        """
        bt_node = getattr(self.blackboard, "bt_node", None)
        wp = getattr(self.blackboard, "current_waypoint", None)
        if bt_node is None or wp is None:
            self.logger.error(f"{self.name}: bt_node 또는 current_waypoint 블랙보드에 없음")
            try:
                self.blackboard.goto_failed = True
            except Exception:
                pass
            return Status.SUCCESS  # 다음 CaptureNode로 넘어가도록

        # 현재 시간
        try:
            now = bt_node.get_clock().now()
            # ROS Time to seconds
            now_sec = now.nanoseconds * 1e-9
        except Exception:
            now = None
            now_sec = time.time()

        # 1) Goal 전송 단계
        if not self.sent:
            # header timestamp 갱신
            try:
                wp.header.stamp = bt_node.get_clock().now().to_msg()
            except Exception:
                pass
            # Nav2 goal 전송
            try:
                send_future = bt_node.send_nav_goal(wp)
            except Exception as e:
                self.logger.error(f"{self.name}: send_nav_goal 예외: {e}")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS
            if send_future is None:
                self.logger.error(f"{self.name}: send_nav_goal 반환 None")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS
            # 전송 성공: acceptance 대기
            self.sent = True
            self.goal_handle_future = send_future
            # start_time 기록
            self.start_time = now_sec
            self.logger.info(f"{self.name}: goal sent, waiting acceptance, start_time={self.start_time:.3f}")
            return Status.RUNNING

        # 2) Acceptance 대기
        if not self.accepted:
            if self.goal_handle_future.done():
                try:
                    goal_handle = self.goal_handle_future.result()
                except Exception as e:
                    self.logger.error(f"{self.name}: goal_handle_future.result() 예외: {e}")
                    goal_handle = None
                if goal_handle is None or not getattr(goal_handle, 'accepted', False):
                    self.logger.error(f"{self.name}: Goal rejected")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
                # Goal accepted
                self.accepted = True
                self.goal_handle = goal_handle
                try:
                    self.result_future = goal_handle.get_result_async()
                except Exception as e:
                    self.logger.error(f"{self.name}: get_result_async 예외: {e}")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
                self.logger.info(f"{self.name}: Goal accepted, waiting result")
            # acceptance 전까지는 RUNNING
            # 또한 timeout 체크: acceptance 기다리다 timeout 초과 시도 가능
            # 여기서는 acceptance 전 타임아웃도 동일하게 처리
            elapsed = now_sec - self.start_time if self.start_time is not None else None
            if elapsed is not None and elapsed > self.max_duration:
                # timeout: cancel goal if possible
                self.logger.warning(f"{self.name}: Acceptance 기다리다 timeout {elapsed:.1f}s 초과, cancel & skip waypoint")
                try:
                    if self.goal_handle is not None:
                        cancel_future = self.goal_handle.cancel_goal_async()
                    # else: 이미 accepted 안 된 상태라 cancel 불필요
                except Exception as e:
                    self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS
            return Status.RUNNING

        # 3) Result 대기
        if self.accepted and self.result_future is not None:
            # timeout 체크
            elapsed = now_sec - self.start_time if self.start_time is not None else None
            if elapsed is not None and elapsed > self.max_duration and not self.arrival_time:
                # 아직 도착 인지 전인데 timeout 초과: cancel & failure 처리
                self.logger.warning(f"{self.name}: 도착 전 timeout {elapsed:.1f}s 초과, cancel & skip waypoint")
                try:
                    # cancel goal
                    self.goal_handle.cancel_goal_async()
                except Exception as e:
                    self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS

            if not self.result_future.done():
                # 아직 결과 안 옴, 계속 RUNNING
                return Status.RUNNING

            # 결과 도착
            try:
                status = self.result_future.result().status
            except Exception as e:
                self.logger.error(f"{self.name}: result_future.result() 예외: {e}")
                status = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                # 도착 성공: arrival_time 기록 후 wait_duration 동안 대기
                if self.arrival_time is None:
                    # 첫 도착 순간
                    if now is not None:
                        self.arrival_time = now
                    else:
                        self.arrival_time = rclpy.time.Time() if hasattr(rclpy.time, 'Time') else None
                    self.logger.info(f"{self.name}: Arrived at waypoint, will wait {self.wait_duration}s before proceeding")
                    return Status.RUNNING
                else:
                    # 대기 중 경과 체크
                    try:
                        curr = bt_node.get_clock().now()
                        elapsed_after = (curr.nanoseconds - self.arrival_time.nanoseconds) * 1e-9
                    except Exception:
                        elapsed_after = None
                    if elapsed_after is None or elapsed_after < self.wait_duration:
                        self.logger.debug(f"{self.name}: Waiting after arrival: elapsed {elapsed_after if elapsed_after else 0:.2f}s")
                        return Status.RUNNING
                    else:
                        self.logger.info(f"{self.name}: Waited {elapsed_after:.2f}s after arrival, proceeding")
                        return Status.SUCCESS
            else:
                # goal 실패 상태 (rejected 아닌 후속 실패)
                self.logger.warning(f"{self.name}: Goal failed with status={status}, skip waypoint")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS

        return Status.RUNNING
