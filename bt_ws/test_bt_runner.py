#!/usr/bin/env python3
# bt_runner.py

import os
import re
import cv2
import numpy as np
import rclpy
import py_trees
from py_trees.common import Status, Access
from geometry_msgs.msg import PoseStamped

import bt_node  # BTNode 클래스가 정의된 모듈
from check_remaining_node import CheckRemaining
from count_node import CountNode
from get_waypoint_node import GetWayPoint
from goto_waypoint_node import GotoWaypoint
from capture_node import CaptureNode
from return_dock_node import ReturnDock

import compare_run_photos  


def monitor_and_compare(photo_dir, logger):


    if not os.path.isabs(photo_dir):
        photo_dir = os.path.expanduser(photo_dir)


    if not os.path.isdir(photo_dir):
        try:
            os.makedirs(photo_dir, exist_ok=True)
            logger.info(f"[Monitor] Created directory: {photo_dir}")
        except Exception as e:
            logger.error(f"[Monitor] Failed to create photo_dir {photo_dir}: {e}")
            return


    try:
        files = os.listdir(photo_dir)
    except Exception as e:
        logger.error(f"[Monitor] Failed to list directory {photo_dir}: {e}")
        return
    logger.debug(f"[Monitor] Current files in {photo_dir}: {files}")


    for fname in files:
        m = re.match(r'waypoint_(\d+)\.png$', fname)
        if not m:
            continue
        idx = m.group(1)

        waypoint_path = os.path.join(photo_dir, fname)
        ref_filename = f"reference{idx}.png"
        ref_path = os.path.join(photo_dir, ref_filename)

        if not os.path.isfile(ref_path):
            logger.warning(f"[Monitor] reference not found for idx={idx}: {ref_path}")
            continue


        out_filename = f"comparison_result_{idx}.png"
        out_path = os.path.join(photo_dir, out_filename)


        if os.path.isfile(out_path):
            logger.debug(f"[Monitor] comparison_result for idx={idx} already exists; skip")
            continue

        ref_img = cv2.imread(ref_path)
        target_img = cv2.imread(waypoint_path)
        if ref_img is None or target_img is None:
            logger.error(f"[Monitor] Failed to load images for idx={idx}: "
                         f"ref_img={'OK' if ref_img is not None else 'None'}, "
                         f"waypoint_img={'OK' if target_img is not None else 'None'}")
            continue

        logger.info(f"[Monitor] Comparing reference{idx}.png <-> waypoint_{idx}.png ...")
        try:
            result = compare_run_photos.detect_changes(ref_img, target_img)
        except Exception as e:
            logger.error(f"[Monitor] detect_changes error for idx={idx}: {e}")
            continue

        try:
            ok = cv2.imwrite(out_path, result)
            if ok:
                logger.info(f"[Monitor] Saved comparison_result_{idx}.png")
            else:
                logger.error(f"[Monitor] Failed to save comparison image for idx={idx}")
        except Exception as e:
            logger.error(f"[Monitor] Error saving comparison image for idx={idx}: {e}")



def display_dashboard(photo_dir, logger):

    if not os.path.isabs(photo_dir):
        photo_dir = os.path.expanduser(photo_dir)
    if not os.path.isdir(photo_dir):
        logger.error(f"[Dashboard] Directory not found: {photo_dir}")
        return


    try:
        files = os.listdir(photo_dir)
    except Exception as e:
        logger.error(f"[Dashboard] Failed to list directory {photo_dir}: {e}")
        return

    idx_list = []
    for fname in files:
        m = re.match(r'comparison_result_(\d+)\.png$', fname)
        if m:
            try:
                idx_list.append(int(m.group(1)))
            except:
                pass
    idx_list = sorted(idx_list)

    if not idx_list:
        logger.warning("[Dashboard] No comparison results to display.")
        return

 
    rows = []
    for idx in idx_list:
        idx_str = str(idx)
        ref_path = os.path.join(photo_dir, f"reference{idx_str}.png")
        curr_path = os.path.join(photo_dir, f"waypoint_{idx_str}.png")
        diff_path = os.path.join(photo_dir, f"comparison_result_{idx_str}.png")

        if not (os.path.isfile(ref_path) and os.path.isfile(curr_path) and os.path.isfile(diff_path)):
            logger.warning(f"[Dashboard] Missing files for idx={idx_str}, skipping row.")
            continue

        ref_img = cv2.imread(ref_path)
        curr_img = cv2.imread(curr_path)
        diff_img = cv2.imread(diff_path)
        if ref_img is None or curr_img is None or diff_img is None:
            logger.error(f"[Dashboard] Failed to load images for idx={idx_str}, skipping.")
            continue

        rows.append((idx_str, ref_img, curr_img, diff_img))

    if not rows:
        logger.warning("[Dashboard] No valid rows to display.")
        return

   
    spacing = 10      
    header_height = 30 
    bg_color = (255, 255, 255)
    label_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2


    processed_rows = []
    for idx_str, ref_img, curr_img, diff_img in rows:
        h_list = [ref_img.shape[0], curr_img.shape[0], diff_img.shape[0]]
        target_h = min(h_list)

        def resize_to_height(img, height):
            h, w = img.shape[:2]
            if h == 0:
                return None
            new_w = int(w * (height / h))
            return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

        ref_r = resize_to_height(ref_img, target_h)
        curr_r = resize_to_height(curr_img, target_h)
        diff_r = resize_to_height(diff_img, target_h)
        if ref_r is None or curr_r is None or diff_r is None:
            continue

        widths = [ref_r.shape[1], curr_r.shape[1], diff_r.shape[1]]
        max_w = max(widths)

        def pad_or_crop_width(img, target_w):
            h, w = img.shape[:2]
            if w == target_w:
                return img
            if w < target_w:
                pad = target_w - w
                return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=bg_color)
            else:
                start = (w - target_w)//2
                return img[:, start:start+target_w]

        ref_c = pad_or_crop_width(ref_r, max_w)
        curr_c = pad_or_crop_width(curr_r, max_w)
        diff_c = pad_or_crop_width(diff_r, max_w)

        processed_rows.append((idx_str, ref_c, curr_c, diff_c))

    if not processed_rows:
        logger.warning("[Dashboard] No rows after resizing.")
        return

    _, sample_ref, sample_curr, sample_diff = processed_rows[0]
    max_w = sample_ref.shape[1]
    total_w = max_w * 3 + spacing * 2

    all_img = None
    vertical_spacer = np.full((spacing, total_w, 3), bg_color, dtype=np.uint8)
    for idx_str, ref_c, curr_c, diff_c in processed_rows:
        header_img = np.full((header_height, total_w, 3), bg_color, dtype=np.uint8)

        ref_x = 0
        curr_x = max_w + spacing
        diff_x = 2 * (max_w + spacing)
        text_y = int(header_height * 0.6)
        cv2.putText(header_img, "Reference", (ref_x + 10, text_y),
                    font, font_scale, label_color, font_thickness, cv2.LINE_AA)
        cv2.putText(header_img, "Current", (curr_x + 10, text_y),
                    font, font_scale, label_color, font_thickness, cv2.LINE_AA)
        cv2.putText(header_img, "Diff", (diff_x + 10, text_y),
                    font, font_scale, label_color, font_thickness, cv2.LINE_AA)
        idx_text = f"Idx {idx_str}"
        (tw, th), _ = cv2.getTextSize(idx_text, font, font_scale, font_thickness)
        cv2.putText(header_img, idx_text, (total_w - tw - 10, text_y),
                    font, font_scale, label_color, font_thickness, cv2.LINE_AA)

        spacer_horiz = np.full((ref_c.shape[0], spacing, 3), bg_color, dtype=np.uint8)
        row_img = np.concatenate([ref_c, spacer_horiz, curr_c, spacer_horiz, diff_c], axis=1)
        combined = np.concatenate([header_img,
                                   vertical_spacer,
                                   row_img], axis=0)
        if all_img is None:
            all_img = combined
        else:
            all_img = np.concatenate([all_img, vertical_spacer, combined], axis=0)

    dashboard_img = all_img

    win_name = "Dashboard All Comparisons"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    h_dash, w_dash = dashboard_img.shape[:2]
    scale = min(1280 / w_dash, 720 / h_dash, 1.0)
    display_img = dashboard_img
    if scale < 1.0:
        display_img = cv2.resize(dashboard_img, (int(w_dash*scale), int(h_dash*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, display_img)

    logger.info("[Dashboard] Press any key or close window to end.")

    while True:
        key = cv2.waitKey(100)
        prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        if prop < 1:
            logger.info("[Dashboard] Window closed by user.")
            break
        if key != -1:
            logger.info(f"[Dashboard] Key pressed (code={key}), closing dashboard.")
            break

    try:
        cv2.destroyWindow(win_name)
    except Exception as e:
        logger.error(f"[Dashboard] Error destroying window: {e}")
    cv2.destroyAllWindows()


def create_behavior_tree():
    
    root = py_trees.composites.Sequence("RootSequence", memory=False)

    fallback = py_trees.composites.Selector("CheckOrProcess", memory=False)
    check_remaining = CheckRemaining("CheckRemaining")
    process_sequence = py_trees.composites.Sequence("ProcessWaypoint", memory=False)
    get_wp = GetWayPoint("GetWayPoint")
    goto_wp = GotoWaypoint("GotoWaypoint")
    capture = CaptureNode("CaptureNode")
    count_after = CountNode("CountAfter")
    process_sequence.add_children([get_wp, goto_wp, capture, count_after])
    fallback.add_children([check_remaining, process_sequence])

    return_dock = ReturnDock("ReturnDock")

    root.add_children([fallback, return_dock])
    return root


def main():
    rclpy.init()
    btnode = bt_node.BTNode()
    logger = btnode.get_logger()

    bb_client = py_trees.blackboard.Client(name="Main")
    bb_client.register_key(key="bt_node", access=Access.WRITE)
    bb_client.bt_node = btnode
    bb_client.register_key(key="initial_pose", access=Access.WRITE)
    bb_client.register_key(key="waypoints", access=Access.WRITE)
    bb_client.register_key(key="current_index", access=Access.WRITE)
    bb_client.register_key(key="photo_paths", access=Access.WRITE)
    bb_client.photo_paths = []  

    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = btnode.get_clock().now().to_msg()
    initial_pose.pose.position.x = -0.3527659295151565
    initial_pose.pose.position.y = -3.292957691502578e-11
    initial_pose.pose.position.z = 0.0
    initial_pose.pose.orientation.x = 0.0
    initial_pose.pose.orientation.y = 0.0
    initial_pose.pose.orientation.z = 0.9981621357977354
    initial_pose.pose.orientation.w = -0.06059992293479631

    btnode.publish_initial_pose(initial_pose)
    try:
        btnode.waitUntilNav2Active()
    except AttributeError:
        pass
    bb_client.initial_pose = initial_pose
    logger.info("Initial pose set and stored on blackboard")

    waypoints = []

    wp1 = PoseStamped()
    wp1.header.frame_id = 'map'
    wp1.header.stamp = btnode.get_clock().now().to_msg()
    wp1.pose.position.x = 4.574370434309949
    wp1.pose.position.y = 5.194985176338236
    wp1.pose.orientation.z = 0.1533114364715118
    wp1.pose.orientation.w = 0.9881779209469526
    waypoints.append(wp1)

    # #wp2 = PoseStamped()
    # wp2.header.frame_id = 'map'
    # wp2.header.stamp = btnode.get_clock().now().to_msg()
    # wp2.pose.position.x = 6.8336787738753957
    # wp2.pose.position.y = -1.7789931122781792
    # wp2.pose.position.z = 0.0
    # wp2.pose.orientation.x = 0.0
    # wp2.pose.orientation.y = 0.0
    # wp2.pose.orientation.z = -0.8518522017756794
    # wp2.pose.orientation.w = 0.5237822317814219
    # waypoints.append(wp2)

    bb_client.waypoints = waypoints
    bb_client.current_index = 0
    logger.info(f"Waypoints set on blackboard: {len(waypoints)} points")

    root = create_behavior_tree()
    tree = py_trees.trees.BehaviourTree(root)
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    logger.info("Setting up BehaviourTree")
    tree.setup(timeout=15.0)

    photo_dir = '/home/choi/waypoint_photos'

    try:
        logger.info("Initial tick of BehaviorTree")
        tree.tick()
        monitor_and_compare(photo_dir, logger)

        while rclpy.ok():
            tree.tick()
            monitor_and_compare(photo_dir, logger)

            rclpy.spin_once(btnode, timeout_sec=0.1)
            if root.status == Status.SUCCESS:
                logger.info("Behavior Tree 전체 완료, 종료합니다.")
                display_dashboard(photo_dir, logger)
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: 중단")
    finally:
        try:
            if hasattr(btnode, 'lifecycleShutdown'):
                btnode.lifecycleShutdown()
            btnode.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
