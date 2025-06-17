#!/usr/bin/env python3
# goto_waypoint_node.py

import py_trees
from py_trees.common import Access, Status
import rclpy
from action_msgs.msg import GoalStatus

class GotoWaypoint(py_trees.behaviour.Behaviour):
    """
    GotoWaypoint:
    - current_waypoint로 Nav2 goal 전송
    - 최초 전송 직후부터 또는 acceptance 받은 시점부터 최대 max_duration 초 동안 대기
    - max_duration 초 초과 시 cancel_goal_async() 호출 시도 후 goto_failed=True 설정, Status.SUCCESS 반환하여 다음 노드로 넘어가도록 함
    - Goal rejected 시 즉시 goto_failed=True, Status.SUCCESS 반환
    """
    def __init__(self, name="GotoWaypoint", max_duration=30.0):
        super(GotoWaypoint, self).__init__(name)
        # 최대 대기 시간 (초)
        self.max_duration = max_duration
        # 상태 트래킹 변수 초기화
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.start_time = None      # rclpy.time.Time 저장
        self.accepted = False       # goal accepted 여부

    def setup(self, timeout=None):
        """
        블랙보드 클라이언트 생성 및 필요한 키 등록
        """
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        # bt_node: Nav2 제어 인터페이스 제공 노드
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        # 현재 waypoint PoseStamped
        self.blackboard.register_key(key="current_waypoint", access=Access.READ)
        # 실패 플래그 기록
        self.blackboard.register_key(key="goto_failed", access=Access.WRITE)
        return True

    def initialise(self):
        """
        매 waypoint마다 초기화.
        sent, accepted, start_time 초기화하고 goto_failed=False로 설정
        """
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.start_time = None
        self.accepted = False
        try:
            # 이전 실패 플래그 리셋
            self.blackboard.goto_failed = False
        except Exception:
            pass
        self.logger.info(f"{self.name}: initialise, attempt waypoint with timeout={self.max_duration}s")

    def update(self):
        """
        - send_nav_goal로 goal 전송 후 max_duration 내에 도착 여부 확인
        - acceptance 전/후 및 result 대기 중 모두 elapsed 체크
        - timeout 시 cancel_goal_async() 호출 시도, goto_failed=True 설정, Status.SUCCESS 반환
        """
        bt_node = getattr(self.blackboard, "bt_node", None)
        wp = getattr(self.blackboard, "current_waypoint", None)
        if bt_node is None or wp is None:
            self.logger.error(f"{self.name}: bt_node 또는 current_waypoint가 블랙보드에 없음")
            try:
                self.blackboard.goto_failed = True
            except Exception:
                pass
            return Status.SUCCESS

        # 1) 현재 ROS 시간 얻기
        try:
            now = bt_node.get_clock().now()  # rclpy.time.Time 객체
        except Exception:
            now = None

        # 2) elapsed 계산 helper: nanoseconds 속성 사용
        def get_elapsed():
            if self.start_time is None or now is None:
                return None
            try:
                # rclpy.time.Time 객체의 nanoseconds 속성: int (전체 나노초)
                delta_ns = now.nanoseconds - self.start_time.nanoseconds
                return delta_ns * 1e-9  # 초 단위 float
            except Exception:
                return None

        elapsed = get_elapsed()
        if elapsed is not None:
            # DEBUG 레벨로 elapsed 로그: tick마다 증가하는지 확인
            self.logger.debug(f"{self.name}: elapsed time = {elapsed:.2f}s")

        # 3) Goal 전송 단계
        if not self.sent:
            # header timestamp 갱신
            try:
                wp.header.stamp = bt_node.get_clock().now().to_msg()
            except Exception:
                pass
            # Nav2 goal 전송 시도 (send_nav_goal 반환 Future라고 가정)
            try:
                send_future = bt_node.send_nav_goal(wp)
            except Exception as e:
                self.logger.error(f"{self.name}: send_nav_goal 예외: {e}")
                send_future = None
            if send_future is None:
                self.logger.error(f"{self.name}: send_nav_goal 실패, skip waypoint")
                try:
                    self.blackboard.goto_failed = True
                except Exception:
                    pass
                return Status.SUCCESS

            # goal 전송 성공: acceptance 대기
            self.sent = True
            self.goal_handle_future = send_future
            # start_time 기록
            if now is not None:
                self.start_time = now
                # nanoseconds를 이용해 로그 남기기
                sec_fp = self.start_time.nanoseconds * 1e-9
                self.logger.debug(f"{self.name}: start_time set to {sec_fp:.3f}s (nanoseconds={self.start_time.nanoseconds})")
            else:
                self.logger.debug(f"{self.name}: start_time 기록 실패 (now=None)")
            self.logger.info(f"{self.name}: goal sent, waiting acceptance (timeout={self.max_duration}s)")
            return Status.RUNNING

        # 4) Acceptance 대기 단계
        if not self.accepted:
            # acceptance 여부 확인
            if self.goal_handle_future.done():
                try:
                    goal_handle = self.goal_handle_future.result()
                except Exception as e:
                    self.logger.error(f"{self.name}: goal_handle_future.result() 예외: {e}")
                    goal_handle = None
                # Goal rejected 또는 handle None 체크
                if goal_handle is None or not getattr(goal_handle, 'accepted', False):
                    self.logger.warning(f"{self.name}: Goal rejected")
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
                # acceptance 받은 직후에도 timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: acceptance 직후 elapsed {elapsed:.2f}s >= timeout, cancel & skip")
                    try:
                        self.goal_handle.cancel_goal_async()
                    except Exception as e:
                        self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
                return Status.RUNNING
            else:
                # 아직 acceptance 못 받은 상태: timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: acceptance 대기 중 elapsed {elapsed:.2f}s >= timeout, skip waypoint")
                    try:
                        if self.goal_handle is not None:
                            self.goal_handle.cancel_goal_async()
                    except Exception as e:
                        self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
                return Status.RUNNING

        # 5) Goal accepted 후 결과 대기 단계
        if self.accepted and self.result_future is not None:
            if self.result_future.done():
                # 결과 도착
                try:
                    status = self.result_future.result().status
                except Exception as e:
                    self.logger.error(f"{self.name}: result_future.result() 예외: {e}")
                    status = None
                if status == GoalStatus.STATUS_SUCCEEDED:
                    self.logger.info(f"{self.name}: 목표 지점 도착 성공")
                    return Status.SUCCESS
                else:
                    self.logger.warning(f"{self.name}: 목표 도착 실패(status={status}), skip waypoint")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
            else:
                # 아직 결과 못 받은 상태: timeout 체크
                if elapsed is not None and elapsed >= self.max_duration:
                    self.logger.warning(f"{self.name}: result 대기 중 elapsed {elapsed:.2f}s >= timeout, cancel & skip")
                    try:
                        self.goal_handle.cancel_goal_async()
                    except Exception as e:
                        self.logger.error(f"{self.name}: cancel_goal_async 예외: {e}")
                    try:
                        self.blackboard.goto_failed = True
                    except Exception:
                        pass
                    return Status.SUCCESS
                return Status.RUNNING

        # 6) 기본: 계속 대기
        return Status.RUNNING
