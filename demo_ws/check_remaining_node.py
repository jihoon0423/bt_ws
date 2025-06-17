#!/usr/bin/env python3
# check_remaining_node.py

import py_trees
from py_trees.common import Access, Status

class CheckRemaining(py_trees.behaviour.Behaviour):
    """
    CheckRemaining:
    - current_index 및 waypoints 길이를 비교하여 남은 waypoint가 없으면 SUCCESS, 남으면 FAILURE
    """
    def __init__(self, name="CheckRemaining"):
        super(CheckRemaining, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        return True

    def initialise(self):
        pass

    def update(self):
        try:
            idx = getattr(self.blackboard, "current_index")
        except Exception:
            idx = None
        waypoints = getattr(self.blackboard, "waypoints", None)
        if idx is None or waypoints is None:
            self.logger.error("CheckRemaining: current_index 또는 waypoints 블랙보드에 없음")
            # 비정상 상황이지만, 실행 흐름을 멈추지 않기 위해 FAILURE로 간주하여 시도 계속
            return Status.FAILURE
        if idx >= len(waypoints):
            self.logger.info(f"CheckRemaining: 더 이상 waypoint 없음 (current_index={idx}, total={len(waypoints)})")
            return Status.SUCCESS
        else:
            self.logger.info(f"CheckRemaining: 남은 waypoint 있음 (current_index={idx}, total={len(waypoints)})")
            return Status.FAILURE
