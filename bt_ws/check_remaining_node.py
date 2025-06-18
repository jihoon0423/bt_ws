#!/usr/bin/env python3
# check_remaining_node.py

import py_trees
from py_trees.common import Access, Status

class CheckRemaining(py_trees.behaviour.Behaviour):
    """
    CheckRemaining: blackboard.current_index와 blackboard.waypoints를 확인하여
    current_index >= len(waypoints)이면 SUCCESS (처리할 waypoint 없음),
    그렇지 않으면 FAILURE (남은 waypoint 있음) 반환.
    """
    def __init__(self, name="CheckRemaining"):
        super(CheckRemaining, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        return True

    def initialise(self):
        pass

    def update(self):
        waypoints = self.blackboard.waypoints  
        idx = self.blackboard.current_index
        if waypoints is None or idx is None:
            self.logger.error("CheckRemaining: waypoints or current_index missing")
            return Status.FAILURE
        if idx >= len(waypoints):
            self.logger.info(f"CheckRemaining: no remaining waypoints (idx={idx}, total={len(waypoints)})")
            return Status.SUCCESS
        else:
            self.logger.info(f"CheckRemaining: remaining waypoints exist (idx={idx}, total={len(waypoints)})")
            return Status.FAILURE
