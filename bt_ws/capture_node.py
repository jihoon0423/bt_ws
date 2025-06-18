#!/usr/bin/env python3
# capture_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import rclpy

class CaptureNode(py_trees.behaviour.Behaviour):

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
        return True

    def initialise(self):
        self.taken = False  

    def update(self):
        if self.taken:
            return Status.SUCCESS

        bt_node = getattr(self.blackboard, "bt_node", None)
        idx = getattr(self.blackboard, "current_index", None)
        if bt_node is None or idx is None:
            self.logger.error("CaptureNode: bt_node or current_index missing")
            return Status.FAILURE

        self.logger.info(f"CaptureNode: waiting {self.wait_before}s before capture")
        rclpy.spin_once(bt_node, timeout_sec=self.wait_before)


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
        if img is None:
            self.logger.warning("CaptureNode: all attempts failed; skipping capture")
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
                self.logger.info(f"CaptureNode: photo_paths already contains {filename}, removing old entry for overwrite semantics")
                paths.remove(filename)
            paths.append(filename)
            self.blackboard.photo_paths = paths
            self.taken = True
            return Status.SUCCESS
        except Exception as e:
            self.logger.error(f"CaptureNode: exception while saving image: {e}")
            self.taken = True
            return Status.FAILURE
