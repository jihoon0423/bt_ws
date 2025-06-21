#!/usr/bin/env python3
# count_node.py

import py_trees
from py_trees.common import Access, Status

class CountNode(py_trees.behaviour.Behaviour):
    """
    CountNode: current_index 증가, goto_failed 리셋
    """
    def __init__(self, name="CountNode"):
        super(CountNode, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="current_index", access=Access.WRITE)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="goto_failed", access=Access.WRITE)
        return True

    def initialise(self):
        pass

    def update(self):
        try:
            idx = getattr(self.blackboard, "current_index")
        except Exception:
            self.logger.error("CountNode: current_index missing")
            return Status.FAILURE
        waypoints = getattr(self.blackboard, "waypoints", None)
        if waypoints is None:
            self.logger.error("CountNode: waypoints missing")
            return Status.FAILURE

        # 다음 인덱스로
        new_idx = idx + 1
        # goto_failed 리셋
        try:
            self.blackboard.goto_failed = False
        except Exception:
            pass

        if new_idx >= len(waypoints):
            self.blackboard.current_index = new_idx
            self.logger.info(f"CountNode: 모든 waypoint 처리 완료 (previous idx={idx}), current_index set to {new_idx}")
            return Status.SUCCESS
        else:
            self.blackboard.current_index = new_idx
            self.logger.info(f"CountNode: current_index 증가 -> {new_idx}")
            return Status.SUCCESS
