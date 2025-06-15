#!/usr/bin/env python3
# dashboard_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import numpy as np
import rclpy

class DashboardNode(py_trees.behaviour.Behaviour):
    """
    Behaviour that displays past and current images along with their difference in a single window.
    """
    def __init__(self, name="DashboardNode"):
        super(DashboardNode, self).__init__(name)
        self.displayed = False
        self.window_opened = False

    def setup(self, timeout=None):
        # Access the blackboard to retrieve stored photo paths (from CaptureNode)
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)       # for ROS node logger (if needed)
        self.blackboard.register_key(key="photo_paths", access=Access.READ)  # list of captured image paths
        return True

    def initialise(self):
        # Reset flags for a new execution cycle
        self.displayed = False
        self.window_opened = False

    def update(self):
        # If we have already displayed and closed the dashboard, we succeed (avoid reopening)
        if self.displayed:
            return Status.SUCCESS

        # Retrieve the list of photo paths from the blackboard
        photo_paths = getattr(self.blackboard, "photo_paths", None) or []
        if not photo_paths:
            self.logger.warning("DashboardNode: photo_paths is empty, nothing to display.")
            self.displayed = True
            return Status.SUCCESS

        # Directory where images are stored (should be ~/waypoint_photos)
        photo_dir = os.path.expanduser('~/waypoint_photos')
        if not os.path.isdir(photo_dir):
            self.logger.error(f"DashboardNode: Directory not found: {photo_dir}")
            self.displayed = True
            return Status.FAILURE

        rows = []  # to collect each horizontal row of images

        # Process each captured photo and its corresponding reference and diff
        for path in photo_paths:
            basename = os.path.basename(path)
            # Expect filenames like "waypoint_<idx>.png"
            try:
                idx_str = basename.split('_')[1].split('.')[0]  # extract the index part
            except Exception as e:
                self.logger.warning(f"DashboardNode: Skipping unexpected filename format: {basename}")
                continue

            ref_path = os.path.join(photo_dir, f"reference{idx_str}.png")
            diff_path = os.path.join(photo_dir, f"comparison_result_{idx_str}.png")
            if not os.path.isfile(ref_path) or not os.path.isfile(diff_path):
                # If either reference or diff image is missing, log and skip this entry
                self.logger.error(f"DashboardNode: Missing reference or diff image for index {idx_str}")
                continue

            # Load images
            ref_img = cv2.imread(ref_path)
            curr_img = cv2.imread(path)         # current waypoint image
            diff_img = cv2.imread(diff_path)    # diff image with bounding boxes
            if ref_img is None or curr_img is None or diff_img is None:
                self.logger.error(f"DashboardNode: Failed to load images for index {idx_str}")
                continue

            # Ensure all images have the same height for horizontal concatenation:contentReference[oaicite:6]{index=6}
            h_ref, w_ref = ref_img.shape[:2]
            h_cur, w_cur = curr_img.shape[:2]
            h_diff, w_diff = diff_img.shape[:2]
            # Determine target height (smallest of the three, to minimize distortion if resizing)
            target_h = min(h_ref, h_cur, h_diff)
            if h_ref != target_h or h_cur != target_h or h_diff != target_h:
                # Resize each image to target_h while preserving aspect ratio
                def resize_height(img, height):
                    h, w = img.shape[:2]
                    if h == 0: 
                        return None
                    new_w = int(w * (height / h))
                    return cv2.resize(img, (new_w, height))
                ref_img = resize_height(ref_img, target_h)
                curr_img = resize_height(curr_img, target_h)
                diff_img = resize_height(diff_img, target_h)
                if ref_img is None or curr_img is None or diff_img is None:
                    self.logger.error(f"DashboardNode: Error resizing images for index {idx_str}")
                    continue
                # Update dimensions after resize
                h_ref, w_ref = ref_img.shape[:2]
                h_cur, w_cur = curr_img.shape[:2]
                h_diff, w_diff = diff_img.shape[:2]

            # Ensure all images have the same width (pad with black borders if needed)
            max_w = max(w_ref, w_cur, w_diff)
            if w_ref != max_w or w_cur != max_w or w_diff != max_w:
                def pad_width(img, target_w):
                    h, w = img.shape[:2]
                    if w >= target_w:
                        return img
                    # Pad the right side with black pixels (BGR=(0,0,0))
                    pad = target_w - w
                    return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                ref_img = pad_width(ref_img, max_w)
                curr_img = pad_width(curr_img, max_w)
                diff_img = pad_width(diff_img, max_w)
                # (Heights remain the same, so no need to update h_ref, etc.)

            # Now concatenate the three images horizontally:contentReference[oaicite:7]{index=7}
            combined_img = np.concatenate((ref_img, curr_img, diff_img), axis=1)
            # Label each section for clarity
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)  # white text for visibility
            cv2.putText(combined_img, f"Reference {idx_str}", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_img, f"Current {idx_str}", (w_ref + 10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_img, f"Diff {idx_str}",    (w_ref + w_cur + 10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
            rows.append(combined_img)

        if not rows:
            self.logger.warning("DashboardNode: No image comparisons available to display.")
            self.displayed = True
            return Status.SUCCESS

        # Stack all rows vertically into one dashboard image (if only one row, use it directly)
        dashboard_img = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)

        # Open a GUI window to show the dashboard image
        if not self.window_opened:
            cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)  # create window that can be resized
            # If the dashboard image is very large, scale it down for display
            max_screen_size = (1280, 720)  # e.g., limit to 1280x720 for viewing
            h_dash, w_dash = dashboard_img.shape[:2]
            scale_w = max_screen_size[0] / w_dash
            scale_h = max_screen_size[1] / h_dash
            scale = min(scale_w, scale_h, 1.0)  # do not upscale, only shrink if needed
            display_img = dashboard_img
            if scale < 1.0:
                new_w = int(w_dash * scale)
                new_h = int(h_dash * scale)
                display_img = cv2.resize(dashboard_img, (new_w, new_h))
            cv2.imshow("Dashboard", display_img)
            self.window_opened = True
            self.logger.info("DashboardNode: Opened dashboard window (press ESC or 'q' to close).")

        # Process events: wait a short time for a key press or window close event:contentReference[oaicite:8]{index=8}
        key = cv2.waitKey(50)  # wait 50 ms for a key event
        # Check for exit conditions: ESC key (27), 'q' key (113), or window manually closed
        if key == 27 or key == ord('q') or cv2.getWindowProperty("Dashboard", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()  # close the window
            self.logger.info("DashboardNode: Dashboard window closed by user.")
            self.displayed = True
            return Status.SUCCESS

        # Continue running (keep window open) until user closes it
        return Status.RUNNING
