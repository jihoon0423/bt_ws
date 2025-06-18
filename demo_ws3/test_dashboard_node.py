# test_dashboard_node.py

import rclpy
import os
import py_trees
from py_trees.blackboard import Blackboard
from dashboard_node import DashboardNode

def main():
    rclpy.init()

    # 1. 이미지 경로 수집
    photo_dir = os.path.expanduser('~/waypoint_photos')
    photo_paths = [
        os.path.join(photo_dir, f)
        for f in os.listdir(photo_dir)
        if f.startswith("waypoint_") and f.endswith(".png")
    ]
    photo_paths.sort()

    # 2. 전역 blackboard에 값 설정
    Blackboard.set("/photo_paths", photo_paths)

    # 3. DashboardNode 실행
    node = DashboardNode()
    node.setup()
    node.initialise()

    while True:
        status = node.update()
        if status != py_trees.common.Status.RUNNING:
            break

if __name__ == "__main__":
    main()
