#!/usr/bin/env python3
# bt_ws/capture_node.py

import os
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import py_trees
from py_trees.common import Access, Status


class CaptureNode(py_trees.behaviour.Behaviour):
    """
    1) 정상 웨이포인트 도달 시 사진 저장
    2) goto_failed=True 로 설정되면 실패 이미지 저장
    """
    def __init__(self,
                 name="CaptureNode",
                 wait_before=0.5,
                 retries=3,
                 retry_interval=0.5,
                 camera_topic="/oakd/rgb/preview/image_raw"):
        super(CaptureNode, self).__init__(name)
        # 정상 캡처 관련
        self.wait_before = wait_before
        self.retries = retries
        self.retry_interval = retry_interval
        self.normal_taken = False

#병합된 코드 이미지 토픽 문자열 수정(검사부탁)
# @@ class CaptureNode(py_trees.behaviour.Behaviour):
# -    def __init__(self,
# -                 name="CaptureNode",
# -                 wait_before=0.5,
# -                 retries=3,
# -                 retry_interval=0.5,
# -                 camera_topic="/camera/color/image_raw"):
# +    def __init__(self,
# +                 name="CaptureNode",
# +                 wait_before=0.5,
# +                 retries=3,
# +                 retry_interval=0.5,
# +                 camera_topic="/camera_front/image_raw"):
#          super(CaptureNode, self).__init__(name)
#          …
#          self.camera_topic = camera_topic

                     
        # 실패 캡처 관련
        self.failure_taken = False

        # 이미지 구독용
        self.bridge = CvBridge()
        self.image_received = False
        self.last_image = None
        self.camera_topic = camera_topic

        # bt_node 에서 이미지 직접 가져오는 대신, 구독된 last_image 사용
        self.bt_node = None

    def setup(self, timeout=None):
        # blackboard 키 등록
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="photo_paths", access=Access.WRITE)
        self.blackboard.register_key(key="goto_failed", access=Access.READ)

        # ROS 2 노드 참조
        if "node" in self.common_setup_kwargs:
            node = self.common_setup_kwargs["node"]
        else:
            # py_trees_ros2 연동 시점에 node 전달받도록 설정하세요
            raise RuntimeError("Node not passed to behaviour setup()")

        # 이미지 토픽 구독
        node.create_subscription(
            Image,
            self.camera_topic,
            self._image_callback,
            10
        )
        return True

    def initialise(self):
        self.normal_taken = False
        self.failure_taken = False
        self.image_received = False
        self.last_image = None

    def update(self):
        # 1) 실패 시점 사진 저장
        goto_failed = getattr(self.blackboard, "goto_failed", False)
        if goto_failed and not self.failure_taken:
            if not self.image_received:
                self.logger.warning(f"{self.name}: Waiting for image to save failure...")
                return Status.RUNNING

            try:
                folder = "failure_images"
                os.makedirs(folder, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(folder, f"failure_{ts}.png")
                cv2.imwrite(fname, self.last_image)
                self.logger.info(f"{self.name}: Saved failure image: {fname}")
                self.failure_taken = True
                # 이후 정상 캡처로 넘어가기 전까지 멈춤
                return Status.SUCCESS
            except Exception as e:
                self.logger.error(f"{self.name}: Error saving failure image - {e}")
                return Status.FAILURE

        # 2) 정상 웨이포인트 사진 저장
        if not self.normal_taken:
            idx = getattr(self.blackboard, "current_index", None)
            if idx is None:
                self.logger.error(f"{self.name}: current_index missing")
                return Status.FAILURE

            if not self.image_received:
                # 아직 이미지가 안 들어왔으면 spin_once 후 대기
                rclpy.spin_once(self.common_setup_kwargs["node"], timeout_sec=self.wait_before)
                if not self.image_received:
                    self.logger.warning(f"{self.name}: No image received yet for waypoint {idx}")
                    return Status.RUNNING

            # 디렉토리 생성
            photo_dir = os.path.expanduser('~/waypoint_photos')
            try:
                os.makedirs(photo_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"{self.name}: Failed to create {photo_dir} - {e}")
                return Status.FAILURE

            # 파일명 및 저장
            filename = os.path.join(photo_dir, f"waypoint_{idx}.png")
            try:
                ok = cv2.imwrite(filename, self.last_image)
                if not ok:
                    raise RuntimeError("cv2.imwrite returned False")
                self.logger.info(f"{self.name}: Saved waypoint image: {filename}")
                # blackboard.photo_paths 업데이트
                paths = getattr(self.blackboard, "photo_paths", []) or []
                if filename in paths:
                    paths.remove(filename)
                paths.append(filename)
                self.blackboard.photo_paths = paths

                self.normal_taken = True
                return Status.SUCCESS
            except Exception as e:
                self.logger.error(f"{self.name}: Error saving waypoint image - {e}")
                return Status.FAILURE

        # 둘 다 완료되었으면 SUCCESS
        return Status.SUCCESS

    def _image_callback(self, msg: Image):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_received = True
        except Exception as e:
            self.logger.error(f"{self.name}: Image conversion failed - {e}")
