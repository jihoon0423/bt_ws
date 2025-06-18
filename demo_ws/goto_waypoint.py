from py_trees.behaviour import Behaviour
from py_trees.common import Status
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.task import Future
from builtin_interfaces.msg import Time as TimeMsg

class GotoWaypoint(Behaviour):
    def __init__(self, name="GotoWaypoint", max_attempts=2):
        super(GotoWaypoint, self).__init__(name)
        self.max_attempts = max_attempts
        self.attempt_count = 0

        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.accepted = False

        self.node = None
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="waypoint", access=self.blackboard.Access.READ)
        self.blackboard.register_key(key="goto_failed", access=self.blackboard.Access.WRITE)

    def setup(self, **kwargs):
        if "node" not in kwargs:
            raise RuntimeError("Node not passed to behaviour setup()")
        self.node = kwargs["node"]
        self.logger.debug(f"{self.name} setup with ROS 2 node")

        self.nav_to_pose_client = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        return True

    def initialise(self):
        self.sent = False
        self.goal_handle = None
        self.goal_handle_future = None
        self.result_future = None
        self.accepted = False
        self.attempt_count += 1

        try:
            self.blackboard.goto_failed = False
        except Exception:
            pass

        self.logger.info(f"{self.name}: attempt {self.attempt_count}/{self.max_attempts}")

    def update(self):
        if self.blackboard is None or self.node is None:
            self.logger.error(f"{self.name}: Blackboard or Node not initialized")
            return Status.FAILURE

        # 1. 목표 좌표 받아오기
        pose = self.blackboard.waypoint
        if pose is None:
            self.logger.error(f"{self.name}: No waypoint set on blackboard")
            return Status.FAILURE

        # 2. 목표 전송
        if not self.sent:
            self.logger.info(f"{self.name}: Sending goal to {pose.pose.position.x}, {pose.pose.position.y}")
            self.sent = True

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose

            try:
                self.goal_handle_future = self.nav_to_pose_client.send_goal_async(goal_msg)
                self.goal_handle_future.add_done_callback(self.goal_response_callback)
            except Exception as e:
                self.logger.error(f"{self.name}: Failed to send goal - {e}")
                return self._handle_failure()

            return Status.RUNNING

        # 3. 목표 수락 대기
        if not self.accepted:
            if self.goal_handle_future is None or not self.goal_handle_future.done():
                return Status.RUNNING

            goal_handle = self.goal_handle_future.result()
            if not goal_handle.accepted:
                self.logger.warning(f"{self.name}: Goal was rejected")
                return self._handle_failure()

            self.logger.info(f"{self.name}: Goal accepted")
            self.accepted = True
            self.goal_handle = goal_handle
            self.result_future = goal_handle.get_result_async()
            return Status.RUNNING

        # 4. 결과 확인
        if self.result_future is None or not self.result_future.done():
            return Status.RUNNING

        result = self.result_future.result()
        if result.status != 4:  # 4 = SUCCEEDED
            self.logger.warning(f"{self.name}: Goal failed with status {result.status}")
            return self._handle_failure()

        self.logger.info(f"{self.name}: Goal succeeded")
        return Status.SUCCESS

    def _handle_failure(self):
        if self.attempt_count >= self.max_attempts:
            self.logger.warning(f"{self.name}: Max attempts reached. Failing.")
            self.blackboard.goto_failed = True
            return Status.SUCCESS
        else:
            self.logger.info(f"{self.name}: Retrying... attempt {self.attempt_count + 1}")
            self.sent = False
            self.accepted = False
            self.goal_handle = None
            self.goal_handle_future = None
            self.result_future = None
            return Status.RUNNING

    def goal_response_callback(self, future: Future):
        try:
            self.goal_handle = future.result()
        except Exception as e:
            self.logger.error(f"{self.name}: Exception in goal response - {e}")
            self.goal_handle = None
