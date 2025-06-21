#!/usr/bin/env python3
# metadata_node.py

import py_trees
from py_trees.common import Access, Status
import os
import json
from datetime import datetime
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose

class MetadataNode(py_trees.behaviour.Behaviour):

    def __init__(self, name="MetadataNode"):
        super(MetadataNode, self).__init__(name)
        self.completed = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="photo_paths", access=Access.READ)
        self.blackboard.register_key(key="current_amcl", access=Access.READ)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        return True

    def initialise(self):
        self.completed = False

    def update(self):
        if self.completed:
            return Status.SUCCESS

        bt_node = getattr(self.blackboard, "bt_node", None)
        try:
            idx = getattr(self.blackboard, "current_index")
        except Exception:
            idx = None

        try:
            photo_paths = getattr(self.blackboard, "photo_paths") or []
        except Exception:
            photo_paths = []

        try:
            waypoints = getattr(self.blackboard, "waypoints") or []
        except Exception:
            waypoints = []

        if bt_node is None or idx is None:
            self.logger.error("MetadataNode: bt_node 또는 current_index 블랙보드에 없음")
            self.completed = True
            return Status.FAILURE

        str_idx = str(idx)
        waypoint_name = f"waypoint_{str_idx}.png"
        reason_name = f"reason_{str_idx}.png"

        matched_path = None
        success = False
        for p in photo_paths:
            try:
                basename = os.path.basename(p)
            except Exception:
                continue
            if basename == waypoint_name:
                matched_path = p
                success = True
                break
            if basename == reason_name:
                if matched_path is None:
                    matched_path = p
                    success = False     
        if matched_path is None:
            return Status.RUNNING


        photo_dir = os.path.expanduser('~/waypoint_project/waypoint_photos')
        timestamp_iso = None
        try:
            now_msg = bt_node.get_clock().now().to_msg()
            sec = now_msg.sec
            nsec = now_msg.nanosec
            dt = datetime.utcfromtimestamp(sec)
            frac = f"{nsec:09d}"
            timestamp_iso = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{frac}Z"
        except Exception as e:
            self.logger.error(f"MetadataNode: timestamp 생성 실패: {e}")
            timestamp_iso = None

        try:
            amcl = getattr(self.blackboard, "current_amcl")
        except Exception:
            amcl = None

        pose_dict = None
        if isinstance(amcl, PoseWithCovarianceStamped):
            p = amcl.pose.pose
            pose_dict = {
                'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
            }
        elif isinstance(amcl, Pose):
            p = amcl
            pose_dict = {
                'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
            }
        else:
            try:
                current = bt_node.get_current_pose()
            except Exception:
                current = None
            if isinstance(current, PoseWithCovarianceStamped):
                p = current.pose.pose
                pose_dict = {
                    'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                    'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
                }
            elif isinstance(current, Pose):
                p = current
                pose_dict = {
                    'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                    'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
                }
            else:
                if 0 <= idx < len(waypoints):
                    wp = waypoints[idx]
                    if isinstance(wp, PoseStamped):
                        p = wp.pose
                        pose_dict = {
                            'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                            'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
                        }
                    else:
                        try:
                            p = wp
                            pose_dict = {
                                'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                                'orientation': {'x': p.orientation.x, 'y': p.orientation.y, 'z': p.orientation.z, 'w': p.orientation.w}
                            }
                        except Exception:
                            pose_dict = None
                else:
                    pose_dict = None

        if os.path.isabs(matched_path):
            image_filename = matched_path
        else:
            image_filename = os.path.join(photo_dir, os.path.basename(matched_path))

        metadata = {
            'waypoint_id': idx,
            'timestamp': timestamp_iso,
            'pose': pose_dict,
            'success': success,
            'image_filename': image_filename
        }


        try:
            os.makedirs(photo_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"MetadataNode: photo_dir 생성 실패: {e}")

        json_filename = os.path.join(photo_dir, f"waypoint_{idx}.json")
        try:
            with open(json_filename, 'w') as jf:
                json.dump(metadata, jf, indent=4, ensure_ascii=False)
            self.logger.info(f"MetadataNode: Saved metadata to {json_filename}: {metadata}")
        except Exception as e:
            self.logger.error(f"MetadataNode: metadata 저장 실패 for idx={idx}: {e}")

        self.completed = True
        return Status.SUCCESS
