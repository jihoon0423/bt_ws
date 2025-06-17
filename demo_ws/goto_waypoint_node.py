import py_trees
from py_trees.common import Access, Status

class GotoWaypoint(py_trees.behaviour.Behaviour):
    def __init__(self, name="GotoWaypoint"):
        super(GotoWaypoint, self).__init__(name)
        self.sent = False
        self.goal_handle = None
        self.result_future = None

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="bt_node", access=Access.READ)
        self.blackboard.register_key(key="current_waypoint", access=Access.READ)
        return True

    def initialise(self):
        self.sent = False
        self.goal_handle = None
        self.result_future = None

    def update(self):
        bt_node = self.blackboard.bt_node
        wp = self.blackboard.current_waypoint
        if bt_node is None or wp is None:
            self.logger.error("GotoWaypoint: bt_node or current_waypoint missing")
            return Status.FAILURE

        if not self.sent:
            wp.header.stamp = bt_node.get_clock().now().to_msg()
            send_future = bt_node.send_nav_goal(wp)
            if send_future is None:
                self.logger.error("GotoWaypoint: send_nav_goal failed")
                return Status.FAILURE
            self.sent = True
            self.logger.info("GotoWaypoint: goal sent, waiting acceptance")
            self.goal_handle_future = send_future
            return Status.RUNNING

        if self.goal_handle is None:
            if self.goal_handle_future.done():
                goal_handle = self.goal_handle_future.result()
                if not goal_handle.accepted:
                    self.logger.error("GotoWaypoint: Goal rejected")
                    return Status.FAILURE
                self.goal_handle = goal_handle
                self.result_future = goal_handle.get_result_async()
                self.logger.info("GotoWaypoint: Goal accepted, waiting result")
                return Status.RUNNING
            else:
                return Status.RUNNING

        if self.result_future is not None:
            if not self.result_future.done():
                return Status.RUNNING
            self.logger.info("GotoWaypoint: result received, assume success")
            return Status.SUCCESS

        return Status.RUNNING
