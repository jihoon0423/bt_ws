#!/usr/bin/env python3
# dashboard_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import numpy as np
import rclpy

class DashboardNode(py_trees.behaviour.Behaviour):
    def __init__(self, name="DashboardNode"):
        super(DashboardNode, self).__init__(name)
        self.displayed = False
        self.window_opened = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="photo_paths", access=Access.READ)
        return True

    def initialise(self):
        self.displayed = False
        self.window_opened = False

    def update(self):
        if self.displayed:
            return Status.SUCCESS

        photo_paths = getattr(self.blackboard, "photo_paths", None) or []
        if not photo_paths:
            self.logger.warning("DashboardNode: photo_paths is empty.")
            self.displayed = True
            return Status.SUCCESS

        photo_dir = os.path.expanduser('~/waypoint_photos')
        if not os.path.isdir(photo_dir):
            self.logger.error(f"DashboardNode: Directory not found: {photo_dir}")
            self.displayed = True
            return Status.FAILURE

        rows = []

        for path in photo_paths:
            basename = os.path.basename(path)
            try:
                idx_str = basename.split('_')[1].split('.')[0]
            except Exception:
                self.logger.warning(f"DashboardNode: Invalid filename: {basename}")
                continue

            ref_path = os.path.join(photo_dir, f"reference{idx_str}.png")
            diff_path = os.path.join(photo_dir, f"comparison_result_{idx_str}.png")
            if not os.path.isfile(ref_path) or not os.path.isfile(diff_path):
                self.logger.error(f"DashboardNode: Missing reference or diff for index {idx_str}")
                continue

            ref_img = cv2.imread(ref_path)
            curr_img = cv2.imread(path)
            diff_img = cv2.imread(diff_path)
            if ref_img is None or curr_img is None or diff_img is None:
                self.logger.error(f"DashboardNode: Failed to load images for {idx_str}")
                continue

            def resize_height(img, height):
                h, w = img.shape[:2]
                if h == 0: return None
                new_w = int(w * (height / h))
                return cv2.resize(img, (new_w, height))

            target_h = min(ref_img.shape[0], curr_img.shape[0], diff_img.shape[0])
            ref_img = resize_height(ref_img, target_h)
            curr_img = resize_height(curr_img, target_h)
            diff_img = resize_height(diff_img, target_h)
            if ref_img is None or curr_img is None or diff_img is None:
                continue

            def pad_width(img, target_w):
                h, w = img.shape[:2]
                if w >= target_w: return img
                pad = target_w - w
                return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])

            max_w = max(ref_img.shape[1], curr_img.shape[1], diff_img.shape[1])
            ref_img = pad_width(ref_img, max_w)
            curr_img = pad_width(curr_img, max_w)
            diff_img = pad_width(diff_img, max_w)

            combined_img = np.concatenate((ref_img, curr_img, diff_img), axis=1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_img, f"Reference {idx_str}", (10, 30), font, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_img, f"Current {idx_str}", (ref_img.shape[1] + 10, 30), font, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_img, f"Diff {idx_str}", (ref_img.shape[1] + curr_img.shape[1] + 10, 30), font, 0.6, (255, 255, 255), 2)

            rows.append(combined_img)

        if not rows:
            self.logger.warning("DashboardNode: No images to display.")
            self.displayed = True
            return Status.SUCCESS

        max_width = max(img.shape[1] for img in rows)
        for i in range(len(rows)):
            h, w = rows[i].shape[:2]
            if w < max_width:
                pad = max_width - w
                rows[i] = cv2.copyMakeBorder(rows[i], 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])

        dashboard_img = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)

        if not self.window_opened:
            cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)
            h_dash, w_dash = dashboard_img.shape[:2]
            scale = min(1280 / w_dash, 720 / h_dash, 1.0)
            display_img = dashboard_img
            if scale < 1.0:
                new_w = int(w_dash * scale)
                new_h = int(h_dash * scale)
                display_img = cv2.resize(dashboard_img, (new_w, new_h))
            cv2.imshow("Dashboard", display_img)
            self.window_opened = True
            self.logger.info("DashboardNode: Dashboard window opened")

        key = cv2.waitKey(50)
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Dashboard", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            self.displayed = True
            self.logger.info("DashboardNode: Closed")
            return Status.SUCCESS

        return Status.RUNNING
