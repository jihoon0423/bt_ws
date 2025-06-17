#!/usr/bin/env python3
# capture_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import rclpy
import math
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped

class CaptureNode(py_trees.behaviour.Behaviour):
    """
    CaptureNode:
    - waypoint에 도착한 후 카메라 이미지를 획득해 저장.
    - 지정된 횟수(retries) 내에 이미지 수신 실패 시 Reason 이미지 생성:
      흰 배경에 빨간 텍스트로 실패 메시지, timestamp, AMCL 위치 정보 포함.
    - 성공/실패 이미지 경로를 블랙보드 photo_paths에 추가.
    """
    def __init__(self, name="CaptureNode", wait_before=0.5, retries=3, retry_interval=0.5):
        super(CaptureNode, self).__init__(name)
        self.wait_before = wait_before
        self.retries = retries
        self.retry_interval = retry_interval
        self.taken = False

    def setup(self, timeout=None):
        # 블랙보드 클라이언트 설정
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        # BTNode 인스턴스를 통해 이미지 획득
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        # 현재 waypoint 인덱스
        self.blackboard.register_key(key="current_index", access=Access.READ)
        # photo_paths 리스트에 WRITE 접근
        self.blackboard.register_key(key="photo_paths", access=Access.WRITE)
        # AMCL pose 정보
        self.blackboard.register_key(key="current_amcl", access=Access.READ)
        return True

    def initialise(self):
        # 매 tick마다 재시도 가능하도록 초기화
        self.taken = False

    def update(self):
        if self.taken:
            return Status.SUCCESS

        bt_node = getattr(self.blackboard, "bt_node", None)
        idx = getattr(self.blackboard, "current_index", None)
        if bt_node is None or idx is None:
            self.logger.error("CaptureNode: bt_node 또는 current_index가 블랙보드에 없음")
            return Status.FAILURE

        # 촬영 전 대기
        self.logger.info(f"CaptureNode: waiting {self.wait_before}s before capture")
        rclpy.spin_once(bt_node, timeout_sec=self.wait_before)

        # 이미지 수신 재시도
        img = None
        for attempt in range(1, self.retries + 1):
            img = bt_node.get_latest_image(timeout_sec=self.retry_interval)
            if img is not None:
                try:
                    h, w = img.shape[:2]
                    self.logger.info(f"CaptureNode: image received on attempt {attempt}, size={w}x{h}")
                except Exception:
                    self.logger.info(f"CaptureNode: image received on attempt {attempt}, but cannot read shape")
                break
            else:
                self.logger.warning(f"CaptureNode: get_latest_image returned None on attempt {attempt}/{self.retries}")

        # 사진 저장 디렉토리
        photo_dir = os.path.expanduser('~/waypoint_project/waypoint_photos')
        try:
            os.makedirs(photo_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"CaptureNode: failed to create directory {photo_dir}: {e}")
            self.taken = True
            return Status.FAILURE

        if img is None:
            # 모든 시도 실패: Reason 이미지 생성
            self.logger.warning(f"CaptureNode: all {self.retries} attempts failed; saving Reason image for idx={idx}")
            reason_filename = os.path.join(photo_dir, f"reason_{idx}.png")
            try:
                self._save_reason_image(reason_filename, idx)
                # 블랙보드 photo_paths 업데이트
                paths = self.blackboard.photo_paths or []
                # 덮어쓰기 고려: 기존에 같은 이름 있으면 제거
                if reason_filename in paths:
                    paths.remove(reason_filename)
                paths.append(reason_filename)
                self.blackboard.photo_paths = paths
                self.logger.info(f"CaptureNode: Saved Reason image to {reason_filename}")
            except Exception as e:
                self.logger.error(f"CaptureNode: exception while saving Reason image: {e}")
            self.taken = True
            # FAILURE 대신 SUCCESS로 처리하여 BT 흐름 계속
            return Status.SUCCESS

        # 정상적으로 이미지 수신된 경우 저장
        filename = os.path.join(photo_dir, f"waypoint_{idx}.png")
        self.logger.info(f"CaptureNode: saving image to {filename}")
        try:
            ok = cv2.imwrite(filename, img)
            if not ok:
                self.logger.error(f"CaptureNode: cv2.imwrite returned False for {filename}")
                self.taken = True
                return Status.FAILURE
            bt_node.get_logger().info(f"CaptureNode: Saved image to {filename}")
            # 블랙보드 photo_paths 업데이트
            paths = self.blackboard.photo_paths or []
            if filename in paths:
                # 덮어쓰기 semantics: 기존 제거
                paths.remove(filename)
            paths.append(filename)
            self.blackboard.photo_paths = paths
            self.taken = True
            return Status.SUCCESS
        except Exception as e:
            self.logger.error(f"CaptureNode: exception while saving image: {e}")
            self.taken = True
            return Status.FAILURE

    def _save_reason_image(self, filepath, idx):
        """
        실패 이유 이미지 생성:
        - 흰 배경
        - 빨간 텍스트로 실패 메시지, timestamp, AMCL 위치 정보 포함
        """
        # 배경 크기 설정 (예: 480x640)
        height = 480
        width = 640
        bg_color = (255, 255, 255)  # 흰 배경
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # 텍스트 스타일
        label_color = (0, 0, 255)  # 빨간색
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 30

        # 첫 줄: 실패 메시지
        text1 = f"Capture failed at idx={idx}"

        # 둘째 줄: timestamp
        text_time = ""
        try:
            now_msg = bt_node = getattr(self.blackboard, "bt_node", None).get_clock().now().to_msg()
            # 초 + 나노초 → ISO 문자열
            sec = now_msg.sec
            nsec_ms = now_msg.nanosec // 1000000
            text_time = f"Time: {sec}.{nsec_ms:03d}"
        except Exception:
            text_time = "Time: unknown"

        # 셋째 줄: AMCL 위치 정보
        pose_text = ""
        amcl = getattr(self.blackboard, "current_amcl", None)
        if isinstance(amcl, PoseWithCovarianceStamped):
            p = amcl.pose.pose
            px = p.position.x
            py = p.position.y
            # orientation에서 yaw 계산 (approx)
            q = p.orientation
            try:
                yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
            except Exception:
                yaw = 0.0
            pose_text = f"AMCL Pose: x={px:.2f}, y={py:.2f}, yaw={yaw:.2f}"
        else:
            pose_text = "AMCL Pose: unknown"

        # 텍스트 그리기 (여유있게 배치)
        y0 = 50
        cv2.putText(img, text1, (30, y0), font, font_scale, label_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text_time, (30, y0 + line_height), font, font_scale, label_color, thickness, cv2.LINE_AA)
        cv2.putText(img, pose_text, (30, y0 + 2*line_height), font, font_scale, label_color, thickness, cv2.LINE_AA)

        # 이미지 저장
        cv2.imwrite(filepath, img)
