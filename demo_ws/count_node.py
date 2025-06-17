#!/usr/bin/env python3
# count_node.py

import py_trees
from py_trees.common import Access, Status

class CountNode(py_trees.behaviour.Behaviour):

    def __init__(self, name="CountNode"):
        super(CountNode, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.WRITE)
        return True

    def initialise(self):
        pass

    def update(self):
        waypoints = self.blackboard.waypoints
        idx = self.blackboard.current_index
        if waypoints is None or idx is None:
            self.logger.error("CountNode: waypoints or current_index missing")
            return Status.FAILURE
        new_idx = idx + 1
        self.blackboard.current_index = new_idx
        self.logger.info(f"CountNode: current_index -> {new_idx}/{len(waypoints)}")
        if new_idx >= len(waypoints):
            return Status.SUCCESS
        else:
            return Status.FAILURE
