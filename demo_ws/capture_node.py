#!/usr/bin/env python3
# capture_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import rclpy
import numpy as np
import math
from geometry_msgs.msg import PoseWithCovarianceStamped

class CaptureNode(py_trees.behaviour.Behaviour):
    """
    CaptureNode:
    - GotoWaypoint가 성공적으로 도달했거나 실패(skip)했을 때 호출
    - goto_failed=True인 경우: Reason 이미지 생성 및 photo_paths에 추가
    - goto_failed=False인 경우: 정상적으로 카메라 이미지를 가져와 저장, photo_paths에 추가
    """
    def __init__(self, name="CaptureNode", wait_before=0.5, retries=3, retry_interval=0.5):
        super(CaptureNode, self).__init__(name)
        self.wait_before = wait_before
        self.retries = retries
        self.retry_interval = retry_interval
        self.taken = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="photo_paths", access=Access.WRITE)
        self.blackboard.register_key(key="current_amcl", access=Access.READ)
        self.blackboard.register_key(key="goto_failed", access=Access.READ)
        # waypoints도 Reason 이미지용 fallback 정보로 읽을 수 있도록
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        return True

    def initialise(self):
        self.taken = False

    def update(self):
        if self.taken:
            return Status.SUCCESS

        bt_node = getattr(self.blackboard, "bt_node", None)
        try:
            idx = getattr(self.blackboard, "current_index")
        except Exception:
            idx = None

        try:
            goto_failed = getattr(self.blackboard, "goto_failed", False)
        except Exception:
            goto_failed = False

        if bt_node is None or idx is None:
            self.logger.error("CaptureNode: bt_node 또는 current_index 블랙보드에 없음")
            self.taken = True
            return Status.SUCCESS

        photo_dir = os.path.expanduser('~/waypoint_photos')
        try:
            os.makedirs(photo_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"CaptureNode: photo_dir 생성 실패: {e}")
            self.taken = True
            return Status.SUCCESS

        if goto_failed:
            # 도달 실패 시 Reason 이미지 생성
            reason_filename = os.path.join(photo_dir, f"reason_{idx}.png")
            self.logger.info(f"CaptureNode: waypoint 도달 실패, Reason 이미지 생성 idx={idx}")
            try:
                self._save_reason_image(reason_filename, idx)
                try:
                    paths = getattr(self.blackboard, "photo_paths") or []
                except Exception:
                    paths = []
                # 중복 방지
                if reason_filename in paths:
                    paths.remove(reason_filename)
                paths.append(reason_filename)
                self.blackboard.photo_paths = paths
                self.logger.info(f"CaptureNode: Saved Reason image to {reason_filename}")
            except Exception as e:
                self.logger.error(f"CaptureNode: Reason 이미지 저장 중 예외: {e}")
            self.taken = True
            return Status.SUCCESS

        # 정상 도달한 경우: 일정 시간 대기 후 이미지 획득 재시도
        self.logger.info(f"CaptureNode: waiting {self.wait_before}s before capture")
        rclpy.spin_once(bt_node, timeout_sec=self.wait_before)

        img = None
        for attempt in range(1, self.retries + 1):
            try:
                img = bt_node.get_latest_image(timeout_sec=self.retry_interval)
            except Exception as e:
                img = None
                self.logger.warning(f"CaptureNode: get_latest_image 예외 시도 {attempt}: {e}")
            if img is not None:
                try:
                    h, w = img.shape[:2]
                    self.logger.info(f"CaptureNode: image received on attempt {attempt}, size={w}x{h}")
                except Exception:
                    self.logger.info(f"CaptureNode: image received on attempt {attempt}, but cannot read shape")
                break
            else:
                self.logger.warning(f"CaptureNode: get_latest_image returned None on attempt {attempt}/{self.retries}")

        if img is None:
            # 모든 시도 실패: Reason 이미지 생성
            self.logger.warning(f"CaptureNode: all {self.retries} attempts failed; saving Reason image idx={idx}")
            reason_filename = os.path.join(photo_dir, f"reason_{idx}.png")
            try:
                self._save_reason_image(reason_filename, idx)
                try:
                    paths = getattr(self.blackboard, "photo_paths") or []
                except Exception:
                    paths = []
                if reason_filename in paths:
                    paths.remove(reason_filename)
                paths.append(reason_filename)
                self.blackboard.photo_paths = paths
                self.logger.info(f"CaptureNode: Saved Reason image to {reason_filename}")
            except Exception as e:
                self.logger.error(f"CaptureNode: Reason 이미지 저장 중 예외: {e}")
            self.taken = True
            return Status.SUCCESS

        # 이미지 획득 성공 시 저장
        filename = os.path.join(photo_dir, f"waypoint_{idx}.png")
        self.logger.info(f"CaptureNode: saving image to {filename}")
        try:
            ok = cv2.imwrite(filename, img)
            if not ok:
                self.logger.error(f"CaptureNode: cv2.imwrite 실패: {filename}")
                self.taken = True
                return Status.SUCCESS
            self.logger.info(f"CaptureNode: Saved image to {filename}")
            try:
                paths = getattr(self.blackboard, "photo_paths") or []
            except Exception:
                paths = []
            if filename in paths:
                paths.remove(filename)
            paths.append(filename)
            self.blackboard.photo_paths = paths
        except Exception as e:
            self.logger.error(f"CaptureNode: 이미지 저장 중 예외: {e}")
        self.taken = True
        return Status.SUCCESS

    def _save_reason_image(self, filepath, idx):
        """
        실패 이유 이미지 생성:
        - 흰 배경에 빨간색 텍스트로 실패 메시지, timestamp, pose 정보 표시
        """
        from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
        height = 480
        width = 640
        bg_color = (255, 255, 255)
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

        label_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 30

        text1 = f"Capture skipped idx={idx}"
        # timestamp
        text_time = ""
        try:
            bt_node = getattr(self.blackboard, "bt_node", None)
            now_msg = bt_node.get_clock().now().to_msg()
            sec = now_msg.sec
            nsec_ms = now_msg.nanosec // 1000000
            text_time = f"Time: {sec}.{nsec_ms:03d}"
        except Exception:
            text_time = "Time: unknown"

        # pose: AMCL이나 waypoints fallback
        pose_text = ""
        try:
            amcl = getattr(self.blackboard, "current_amcl", None)
        except Exception:
            amcl = None
        if isinstance(amcl, PoseWithCovarianceStamped):
            p = amcl.pose.pose
            px = p.position.x
            py = p.position.y
            try:
                q = p.orientation
                yaw = math.atan2(2.0*(q.w*q.z), 1.0 - 2.0*(q.z*q.z))
            except Exception:
                yaw = 0.0
            pose_text = f"AMCL Pose: x={px:.2f}, y={py:.2f}, yaw={yaw:.2f}"
        else:
            # fallback to waypoints[current_index]
            try:
                idx_bb = getattr(self.blackboard, "current_index")
                waypoints = getattr(self.blackboard, "waypoints", None)
                if waypoints and 0 <= idx_bb < len(waypoints):
                    wp = waypoints[idx_bb]
                    p = wp.pose
                    pose_text = f"WP Pose: x={p.position.x:.2f}, y={p.position.y:.2f}"
                else:
                    pose_text = "Pose unknown"
            except Exception:
                pose_text = "Pose unknown"

        y0 = 50
        cv2.putText(img, text1, (30, y0), font, font_scale, label_color, thickness, cv2.LINE_AA)
        cv2.putText(img, text_time, (30, y0 + line_height), font, font_scale, label_color, thickness, cv2.LINE_AA)
        cv2.putText(img, pose_text, (30, y0 + 2*line_height), font, font_scale, label_color, thickness, cv2.LINE_AA)

        cv2.imwrite(filepath, img)
