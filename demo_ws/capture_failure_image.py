from py_trees.behaviour import Behaviour
from py_trees.common import Status
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from datetime import datetime

class CaptureFailureImage(Behaviour):
    def __init__(self, name="CaptureFailureImage"):
        super(CaptureFailureImage, self).__init__(name)
        self.node = None
        self.image_received = False
        self.bridge = CvBridge()
        self.last_image = None

        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="goto_failed", access=self.blackboard.Access.READ)

    def setup(self, **kwargs):
        if "node" not in kwargs:
            raise RuntimeError("Node not passed to behaviour setup()")
        self.node = kwargs["node"]
        self.logger.debug(f"{self.name} setup with ROS 2 node")

        self.node.create_subscription(
            Image,
            "/camera/color/image_raw",  # 적절한 토픽으로 변경
            self.image_callback,
            10
        )

        return True

    def initialise(self):
        self.image_received = False
        self.last_image = None

    def update(self):
        # 실패 상태가 아니라면 아무 작업 안 함
        if not self.blackboard.goto_failed:
            self.logger.debug(f"{self.name}: No failure detected, skipping.")
            return Status.SUCCESS

        if not self.image_received or self.last_image is None:
            self.logger.warning(f"{self.name}: No image received yet.")
            return Status.RUNNING

        try:
            # 디렉토리 생성
            folder = "failure_images"
            os.makedirs(folder, exist_ok=True)

            # 타임스탬프 기반 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder, f"failure_{timestamp}.png")

            # 저장
            cv2.imwrite(filename, self.last_image)
            self.logger.info(f"{self.name}: Saved failure image to {filename}")
        except Exception as e:
            self.logger.error(f"{self.name}: Error saving image - {e}")
            return Status.FAILURE

        return Status.SUCCESS

    def image_callback(self, msg: Image):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_received = True
        except Exception as e:
            self.logger.error(f"{self.name}: Failed to convert image - {e}")
