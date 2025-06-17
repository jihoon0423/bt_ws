#!/usr/bin/env python3
# metadata_node.py

import py_trees
from py_trees.common import Access, Status
import os
import json
from datetime import datetime
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose

class MetadataNode(py_trees.behaviour.Behaviour):
    """
    MetadataNode:
    - current_index 기준으로 블랙보드 photo_paths에 waypoint_<idx>.png 또는 reason_<idx>.png가 나타날 때까지 대기
    - AMCL pose(current_amcl)가 없으면 블랙보드에 저장된 waypoints[current_index]의 pose를 사용
    - BTNode clock 및 pose 정보를 합쳐 metadata 구성
    - ~/waypoint_photos/waypoint_<idx>.json 으로 저장
    """
    def __init__(self, name="MetadataNode"):
        super(MetadataNode, self).__init__(name)
        self.completed = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        # BTNode 인스턴스 접근
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        # 현재 waypoint 인덱스
        self.blackboard.register_key(key="current_index", access=Access.READ)
        # waypoints 리스트 (PoseStamped 객체들)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        # photo_paths 리스트: CaptureNode나 Reason 이미지 생성 시 여기에 추가됨
        self.blackboard.register_key(key="photo_paths", access=Access.READ)
        # AMCL pose 정보
        self.blackboard.register_key(key="current_amcl", access=Access.READ)
        return True

    def initialise(self):
        self.completed = False

    def update(self):
        if self.completed:
            return Status.SUCCESS

        # 블랙보드 안전 접근
        bt_node = getattr(self.blackboard, "bt_node", None)
        try:
            idx = getattr(self.blackboard, "current_index")
        except KeyError:
            idx = None

        try:
            photo_paths = getattr(self.blackboard, "photo_paths") or []
        except KeyError:
            photo_paths = []

        try:
            waypoints = getattr(self.blackboard, "waypoints") or []
        except KeyError:
            waypoints = []

        if bt_node is None or idx is None:
            self.logger.error("MetadataNode: bt_node 또는 current_index가 블랙보드에 없음")
            self.completed = True
            return Status.FAILURE

        # photo_paths에서 현재 idx 관련 파일이 등장했는지 확인
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
                # waypoint 이미지도 있는지 계속 탐색

        if matched_path is None:
            # 아직 사진이 찍히지 않았거나 Reason 이미지도 생성되지 않음: 계속 대기
            return Status.RUNNING

        # photo_dir: CaptureNode와 동일하게 사용
        photo_dir = os.path.expanduser('~/waypoint_photos')

        # timestamp: BTNode clock 사용, nanosecond까지 포함한 ISO 8601 형식
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

        # AMCL pose 읽기
        try:
            amcl = getattr(self.blackboard, "current_amcl")
        except KeyError:
            amcl = None

        pose_dict = None
        # 1) AMCL pose가 유효한 경우
        if isinstance(amcl, PoseWithCovarianceStamped):
            p = amcl.pose.pose
            pose_dict = {
                'position': {
                    'x': p.position.x,
                    'y': p.position.y,
                    'z': p.position.z
                },
                'orientation': {
                    'x': p.orientation.x,
                    'y': p.orientation.y,
                    'z': p.orientation.z,
                    'w': p.orientation.w
                }
            }
        elif isinstance(amcl, Pose):
            p = amcl
            pose_dict = {
                'position': {
                    'x': p.position.x,
                    'y': p.position.y,
                    'z': p.position.z
                },
                'orientation': {
                    'x': p.orientation.x,
                    'y': p.orientation.y,
                    'z': p.orientation.z,
                    'w': p.orientation.w
                }
            }
        else:
            # 2) AMCL pose가 없으면 bt_node.get_current_pose() 시도
            try:
                current = bt_node.get_current_pose()
            except Exception:
                current = None

            if isinstance(current, PoseWithCovarianceStamped):
                p = current.pose.pose
                pose_dict = {
                    'position': {
                        'x': p.position.x,
                        'y': p.position.y,
                        'z': p.position.z
                    },
                    'orientation': {
                        'x': p.orientation.x,
                        'y': p.orientation.y,
                        'z': p.orientation.z,
                        'w': p.orientation.w
                    }
                }
            elif isinstance(current, Pose):
                p = current
                pose_dict = {
                    'position': {
                        'x': p.position.x,
                        'y': p.position.y,
                        'z': p.position.z
                    },
                    'orientation': {
                        'x': p.orientation.x,
                        'y': p.orientation.y,
                        'z': p.orientation.z,
                        'w': p.orientation.w
                    }
                }
            else:
                # 3) AMCL 정보가 전혀 없으면, 블랙보드 waypoints 리스트에서 미리 지정한 좌표 사용
                # waypoints[idx]가 PoseStamped일 것으로 가정
                if 0 <= idx < len(waypoints):
                    wp = waypoints[idx]
                    if isinstance(wp, PoseStamped):
                        p = wp.pose
                        pose_dict = {
                            'position': {
                                'x': p.position.x,
                                'y': p.position.y,
                                'z': p.position.z
                            },
                            'orientation': {
                                'x': p.orientation.x,
                                'y': p.orientation.y,
                                'z': p.orientation.z,
                                'w': p.orientation.w
                            }
                        }
                    else:
                        # PoseStamped 이외 타입이라면 시도
                        try:
                            # 예: Pose 객체일 수도 있으나, get_waypoint_node에서 블랙보드에 PoseStamped를 넣었다고 가정
                            p = wp
                            pose_dict = {
                                'position': {
                                    'x': p.position.x,
                                    'y': p.position.y,
                                    'z': p.position.z
                                },
                                'orientation': {
                                    'x': p.orientation.x,
                                    'y': p.orientation.y,
                                    'z': p.orientation.z,
                                    'w': p.orientation.w
                                }
                            }
                        except Exception:
                            pose_dict = None
                else:
                    pose_dict = None

        # image_filename: matched_path 절대 경로로
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

        # JSON 파일로 저장
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
