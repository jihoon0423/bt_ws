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


        spacing = 10         
        header_height = 30    
        bg_color = (255,255,255)
        label_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        rows = []
        def extract_idx(p):
            name = os.path.basename(p)
            try:
                idx = int(name.split('_')[1].split('.')[0])
                return idx
            except:
                return float('inf')
        photo_paths_sorted = sorted(photo_paths, key=extract_idx)

        for path in photo_paths_sorted:
            basename = os.path.basename(path)
            try:
                idx_str = basename.split('_')[1].split('.')[0]
            except Exception:
                self.logger.warning(f"DashboardNode: Invalid filename: {basename}")
                continue

            ref_path = os.path.join(photo_dir, f"reference{idx_str}.png")
            diff_path = os.path.join(photo_dir, f"comparison_result_{idx_str}.png")
            curr_path = path  

            if not os.path.isfile(ref_path):
                self.logger.error(f"DashboardNode: Missing reference for index {idx_str}: {ref_path}")
                continue
            if not os.path.isfile(diff_path):
                self.logger.error(f"DashboardNode: Missing diff for index {idx_str}: {diff_path}")
                continue

            ref_img = cv2.imread(ref_path)
            curr_img = cv2.imread(curr_path)
            diff_img = cv2.imread(diff_path)
            if ref_img is None or curr_img is None or diff_img is None:
                self.logger.error(f"DashboardNode: Failed to load images for index {idx_str}")
                continue

            def resize_to_height(img, height):
                h, w = img.shape[:2]
                if h == 0:
                    return None
                new_w = int(w * (height / h))
                return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

            def pad_or_crop_width(img, target_w):
                h, w = img.shape[:2]
                if w == target_w:
                    return img
                if w < target_w:
                    pad = target_w - w
                    return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=bg_color)
                else:
                    start = (w - target_w) // 2
                    return img[:, start:start+target_w]

            target_h = min(ref_img.shape[0], curr_img.shape[0], diff_img.shape[0])
            ref_r = resize_to_height(ref_img, target_h)
            curr_r = resize_to_height(curr_img, target_h)
            diff_r = resize_to_height(diff_img, target_h)
            if ref_r is None or curr_r is None or diff_r is None:
                continue

            widths = [ref_r.shape[1], curr_r.shape[1], diff_r.shape[1]]
            max_w = max(widths)
            ref_c = pad_or_crop_width(ref_r, max_w)
            curr_c = pad_or_crop_width(curr_r, max_w)
            diff_c = pad_or_crop_width(diff_r, max_w)

            spacer_horiz = np.full((target_h, spacing, 3), bg_color, dtype=np.uint8)
            row_img = np.concatenate([ref_c, spacer_horiz, curr_c, spacer_horiz, diff_c], axis=1)
            rows.append((idx_str, row_img))

        if not rows:
            self.logger.warning("DashboardNode: No valid image rows to display.")
            self.displayed = True
            return Status.SUCCESS


        row_widths = [img.shape[1] for (_, img) in rows]
        total_w = max(row_widths)
        for i, (idx_str, img) in enumerate(rows):
            h, w = img.shape[:2]
            if w != total_w:
                if w < total_w:
                    pad = total_w - w
                    img2 = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=bg_color)
                else:
                    start = (w - total_w)//2
                    img2 = img[:, start:start+total_w]
                rows[i] = (idx_str, img2)

        all_rows_img = None
        vertical_spacer = np.full((spacing, total_w, 3), bg_color, dtype=np.uint8)

        for idx_str, row_img in rows:
            header_img = np.full((header_height, total_w, 3), bg_color, dtype=np.uint8)
            max_w_calc = (total_w - 2*spacing) // 3
            ref_x = 0
            curr_x = max_w_calc + spacing
            diff_x = (max_w_calc + spacing) * 2
            text_y = int(header_height * 0.6)
            cv2.putText(header_img, "Reference", (ref_x + 10, text_y),
                        font, font_scale, label_color, font_thickness, cv2.LINE_AA)
            cv2.putText(header_img, "Current", (curr_x + 10, text_y),
                        font, font_scale, label_color, font_thickness, cv2.LINE_AA)
            cv2.putText(header_img, "Diff", (diff_x + 10, text_y),
                        font, font_scale, label_color, font_thickness, cv2.LINE_AA)

            combined = header_img
            combined = np.concatenate([combined, vertical_spacer, row_img], axis=0)

            if all_rows_img is None:
                all_rows_img = combined
            else:
                all_rows_img = np.concatenate([all_rows_img, vertical_spacer, combined], axis=0)

        dashboard_img = all_rows_img

        if not self.window_opened:
            cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)
            h_dash, w_dash = dashboard_img.shape[:2]
            scale = min(1280 / w_dash, 720 / h_dash, 1.0)
            display_img = dashboard_img
            if scale < 1.0:
                new_w = int(w_dash * scale)
                new_h = int(h_dash * scale)
                display_img = cv2.resize(dashboard_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
