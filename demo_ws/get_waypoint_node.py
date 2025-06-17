import py_trees
from py_trees.common import Access, Status

class GetWayPoint(py_trees.behaviour.Behaviour):
    def __init__(self, name="GetWayPoint"):
        super(GetWayPoint, self).__init__(name)

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        self.blackboard.register_key(key="current_waypoint", access=Access.WRITE)
        return True

    def initialise(self):
        pass

    def update(self):
        waypoints = self.blackboard.waypoints
        idx = self.blackboard.current_index
        if waypoints is None or idx is None:
            self.logger.error("GetWayPoint: waypoints or current_index missing")
            return Status.FAILURE
        if idx < 0 or idx >= len(waypoints):
            self.logger.error(f"GetWayPoint: index {idx} out of range (len={len(waypoints)})")
            return Status.FAILURE
        wp = waypoints[idx]
        bt_node = getattr(self.blackboard, "bt_node", None)
        if bt_node is not None:
            wp.header.stamp = bt_node.get_clock().now().to_msg()
        self.blackboard.current_waypoint = wp
        self.logger.info(f"GetWayPoint: selected waypoint index {idx}: ({wp.pose.position.x}, {wp.pose.position.y})")
        return Status.SUCCESS
