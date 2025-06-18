#!/usr/bin/env python3
# capture_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import rclpy
import time

class CaptureNode(py_trees.behaviour.Behaviour):
    """
    CaptureNode:
    - GotoWaypoint에서 설정한 blackboard.goto_failed 여부에 따라:
      * 정상 도착 시: waypoint_{idx}.png로 저장
      * 실패 타임아웃 시: 현재 위치에서 사진 찍고 reason_{idx}.png로 저장 (약간 대기 후)
    """
    def __init__(self, name="CaptureNode", wait_before=0.5, retries=3, retry_interval=0.5, fail_wait=2.0):
        """
        :param wait_before: 정상 촬영 전 기본 대기 시간(초)
        :param retries: 시도 횟수
        :param retry_interval: 각 시도 간 대기 시간(초)
        :param fail_wait: 실패 타임아웃 시 촬영 전 대기 시간(초)  (예: 2초)
        """
        super(CaptureNode, self).__init__(name)
        self.wait_before = wait_before
        self.retries = retries
        self.retry_interval = retry_interval
        self.fail_wait = fail_wait
        self.taken = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="photo_paths", access=Access.WRITE)
        # 실패 플래그 읽기
        self.blackboard.register_key(key="goto_failed", access=Access.READ)
        return True

    def initialise(self):
        self.taken = False

    def update(self):
        if self.taken:
            return Status.SUCCESS

        bt_node = getattr(self.blackboard, "bt_node", None)
        idx = getattr(self.blackboard, "current_index", None)
        goto_failed = getattr(self.blackboard, "goto_failed", False)
        if bt_node is None or idx is None:
            self.logger.error("CaptureNode: bt_node or current_index missing")
            return Status.FAILURE

        if goto_failed:
            # 실패 타임아웃 상태: 약간 대기 후 사진 찍기
            self.logger.info(f"CaptureNode: waypoint {idx} 도착 실패 후 사진 촬영 준비: wait {self.fail_wait}s")
            # ROS spin 또는 sleep 방식: bt_node를 spin하여 콜백 처리
            try:
                # wait_fail 동안 spin once 반복
                start = time.time()
                while time.time() - start < self.fail_wait:
                    rclpy.spin_once(bt_node, timeout_sec=0.1)
                # 이후 이미지 받아오기
            except Exception as e:
                self.logger.error(f"CaptureNode: fail_wait spin 예외: {e}")

            # 시도하여 이미지 얻기
            img = None
            for attempt in range(1, self.retries + 1):
                img = bt_node.get_latest_image(timeout_sec=self.retry_interval)
                if img is not None:
                    self.logger.info(f"CaptureNode: (failure) image received on attempt {attempt}, idx={idx}")
                    break
                else:
                    self.logger.warning(f"CaptureNode: (failure) get_latest_image None on attempt {attempt}/{self.retries}")
            if img is None:
                self.logger.warning(f"CaptureNode: (failure) all attempts failed for idx={idx}, skipping capture")
                # 사진 못 찍어도 다음으로
                self.taken = True
                return Status.SUCCESS

            # 저장: reason_{idx}.png
            photo_dir = os.path.expanduser('~/waypoint_photos')
            try:
                os.makedirs(photo_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"CaptureNode: failed to create directory {photo_dir}: {e}")
                self.taken = True
                return Status.FAILURE

            filename = os.path.join(photo_dir, f"reason_{idx}.png")
            self.logger.info(f"CaptureNode: saving failure image to {filename}")
            try:
                ok = cv2.imwrite(filename, img)
                if not ok:
                    self.logger.error(f"CaptureNode: cv2.imwrite False for {filename}")
                    self.taken = True
                    return Status.FAILURE
                bt_node.get_logger().info(f"CaptureNode: Saved failure image to {filename}")
                paths = self.blackboard.photo_paths or []
                # 실패 이미지는 photo_paths에 포함시키되, 기존 waypoint 이미지와 구분
                if filename in paths:
                    paths.remove(filename)
                paths.append(filename)
                self.blackboard.photo_paths = paths
                self.taken = True
                return Status.SUCCESS
            except Exception as e:
                self.logger.error(f"CaptureNode: exception while saving failure image: {e}")
                self.taken = True
                return Status.FAILURE

        else:
            # 정상 도착 시: 기존 로직 (waypoint_{idx}.png 저장)
            self.logger.info(f"CaptureNode: waiting {self.wait_before}s before capture for idx={idx}")
            try:
                rclpy.spin_once(bt_node, timeout_sec=self.wait_before)
            except Exception:
                pass

            img = None
            for attempt in range(1, self.retries + 1):
                img = bt_node.get_latest_image(timeout_sec=self.retry_interval)
                if img is not None:
                    try:
                        h, w = img.shape[:2]
                        self.logger.info(f"CaptureNode: image received on attempt {attempt}, size={w}x{h}, idx={idx}")
                    except Exception:
                        self.logger.info(f"CaptureNode: image received on attempt {attempt}, idx={idx}")
                    break
                else:
                    self.logger.warning(f"CaptureNode: get_latest_image None on attempt {attempt}/{self.retries}")
            if img is None:
                self.logger.warning(f"CaptureNode: all attempts failed; skipping capture for idx={idx}")
                self.taken = True
                return Status.SUCCESS

            photo_dir = os.path.expanduser('~/waypoint_photos')
            try:
                os.makedirs(photo_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"CaptureNode: failed to create directory {photo_dir}: {e}")
                self.taken = True
                return Status.FAILURE

            filename = os.path.join(photo_dir, f"waypoint_{idx}.png")
            self.logger.info(f"CaptureNode: saving image to {filename}")
            try:
                ok = cv2.imwrite(filename, img)
                if not ok:
                    self.logger.error(f"CaptureNode: cv2.imwrite returned False for {filename}")
                    self.taken = True
                    return Status.FAILURE
                bt_node.get_logger().info(f"CaptureNode: Saved image to {filename}")
                paths = self.blackboard.photo_paths or []
                if filename in paths:
                    paths.remove(filename)
                paths.append(filename)
                self.blackboard.photo_paths = paths
                self.taken = True
                return Status.SUCCESS
            except Exception as e:
                self.logger.error(f"CaptureNode: exception while saving image: {e}")
                self.taken = True
                return Status.FAILURE
