# has_remaining_node.py

import py_trees
from py_trees.common import Access, Status

class HasRemaining(py_trees.behaviour.Behaviour):
    """
    current_index < len(waypoints)이면 SUCCESS (남은 웨이포인트 있음)
    그렇지 않으면 FAILURE (모두 처리됨)
    """
    def __init__(self, name="HasRemaining"):
        super(HasRemaining, self).__init__(name)
        self.blackboard = None

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="waypoints", access=Access.READ)
        self.blackboard.register_key(key="current_index", access=Access.READ)
        return True

    def update(self):
        waypoints = getattr(self.blackboard, "waypoints", None)
        idx = getattr(self.blackboard, "current_index", None)
        if waypoints is None or idx is None:
            self.logger.error(f"{self.name}: waypoints or current_index missing on blackboard")
            # 웨이포인트 정보가 없으면 반복 루프를 멈추도록 FAILURE 반환
            return Status.FAILURE
        total = len(waypoints)
        self.logger.info(f"{self.name}: current_index={idx}, total={total}")
        if idx < total:
            self.logger.debug(f"{self.name}: 남은 웨이포인트 있음 → SUCCESS")
            return Status.SUCCESS
        else:
            self.logger.info(f"{self.name}: 모든 웨이포인트 처리 완료 → FAILURE")
            return Status.FAILURE
