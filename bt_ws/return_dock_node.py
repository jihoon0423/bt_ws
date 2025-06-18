#!/usr/bin/env python3
# return_dock_node.py

import py_trees
from py_trees.common import Access, Status

class ReturnDock(py_trees.behaviour.Behaviour):
    """
    ReturnDock: blackboard.initial_pose로 Nav2 이동.
    """
    def __init__(self, name="ReturnDock"):
        super(ReturnDock, self).__init__(name)
        self.sent = False
        self.send_goal_future = None
        self.goal_handle = None
        self.result_future = None

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="initial_pose", access=Access.READ)
        return True

    def initialise(self):
        self.sent = False
        self.send_goal_future = None
        self.goal_handle = None
        self.result_future = None

    def update(self):
        bt_node = self.blackboard.bt_node
        initial_pose = self.blackboard.initial_pose
        if bt_node is None or initial_pose is None:
            self.logger.error("ReturnDock: bt_node or initial_pose missing")
            return Status.FAILURE

        if not self.sent:
            initial_pose.header.stamp = bt_node.get_clock().now().to_msg()
            self.send_goal_future = bt_node.send_nav_goal(initial_pose)
            if self.send_goal_future is None:
                self.logger.error("ReturnDock: send_nav_goal failed")
                return Status.FAILURE
            self.sent = True
            self.logger.info("ReturnDock: return-to-dock goal sent")
            return Status.RUNNING

        if self.goal_handle is None:
            if self.send_goal_future.done():
                goal_handle = self.send_goal_future.result()
                if not goal_handle.accepted:
                    self.logger.error("ReturnDock: Goal rejected")
                    return Status.FAILURE
                self.goal_handle = goal_handle
                self.result_future = goal_handle.get_result_async()
                self.logger.info("ReturnDock: Goal accepted, waiting result")
                return Status.RUNNING
            else:
                return Status.RUNNING

        if self.result_future is not None:
            if not self.result_future.done():
                return Status.RUNNING
            self.logger.info("ReturnDock: Arrived at dock (assume success)")
            return Status.SUCCESS

        return Status.RUNNING
