#!/usr/bin/env python3
# get_waypoint_node.py

import py_trees
from py_trees.common import Access, Status

class GetWayPoint(py_trees.behaviour.Behaviour):
    """
    GetWayPoint:
    - current_index에 해당하는 waypoints 리스트의 PoseStamped를 읽어서 current_waypoint로 블랙보드에 저장
    - 새로운 waypoint 선택 시 goto_failed=False로 리셋
    """
    def __init__(self, name="GetWayPoint"):
        super(GetWayPoint, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="current_waypoint", access=Access.WRITE)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="goto_failed", access=Access.WRITE)
        return True

    def initialise(self):
        pass

    def update(self):
        waypoints = getattr(self.blackboard, "waypoints", None)
        idx = getattr(self.blackboard, "current_index", None)
        if waypoints is None or idx is None:
            self.logger.error("GetWayPoint: waypoints 또는 current_index 블랙보드에 없음")
            return Status.FAILURE
        if not isinstance(waypoints, list) or idx < 0 or idx >= len(waypoints):
            self.logger.error(f"GetWayPoint: index {idx} out of range (len={len(waypoints) if waypoints else 'None'})")
            return Status.FAILURE
        wp = waypoints[idx]
        bt_node = getattr(self.blackboard, "bt_node", None)
        if bt_node is not None:
            try:
                wp.header.stamp = bt_node.get_clock().now().to_msg()
            except Exception:
                pass
        self.blackboard.current_waypoint = wp
        # 새 waypoint 선택 시 goto_failed 리셋
        try:
            self.blackboard.goto_failed = False
        except Exception:
            pass
        self.logger.info(f"GetWayPoint: selected waypoint index {idx}: ({wp.pose.position.x}, {wp.pose.position.y})")
        return Status.SUCCESS
