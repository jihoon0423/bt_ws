#!/usr/bin/env python3
# bt_node.py

import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading

class BTNode(Node):
    def __init__(self):
        super().__init__('bt_node')

        action_name = 'navigate_to_pose'  
        self._nav_action_client = ActionClient(self, NavigateToPose, action_name)
        self.get_logger().info(f"Waiting for action server '{action_name}'...")
        if not self._nav_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"NavigateToPose action server '{action_name}' not available!")


        self._initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)


        self.bridge = CvBridge()
        self.latest_image = None
        self._image_lock = threading.Lock()


        camera_topic = '/oakd/rgb/preview/image_raw'
        try:
            self.create_subscription(Image, camera_topic, self._image_callback, 10)
            self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")
        except Exception as e:
            self.get_logger().warning(f"Failed to subscribe to camera topic '{camera_topic}': {e}")




    def _image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"_image_callback cv_bridge error: {e}")
            return
        with self._image_lock:
            self.latest_image = cv_image.copy()

    def get_latest_image(self, timeout_sec=5.0):
        elapsed = 0.0
        interval = 0.1
        while elapsed < timeout_sec:
            with self._image_lock:
                img = None if self.latest_image is None else self.latest_image.copy()
            if img is not None:
                return img
            rclpy.spin_once(self, timeout_sec=interval)
            elapsed += interval
        self.get_logger().warning(f"get_latest_image: no image after {timeout_sec}s")
        return None


    
    def publish_initial_pose(self, pose: PoseStamped, covariance=None):
        pwc = PoseWithCovarianceStamped()
        pwc.header.stamp = self.get_clock().now().to_msg()
        pwc.header.frame_id = pose.header.frame_id
        pwc.pose.pose = pose.pose
        if covariance and len(covariance) == 36:
            pwc.pose.covariance = covariance
        else:
            pwc.pose.covariance = [0.0]*36
        self._initialpose_pub.publish(pwc)
        self.get_logger().info("Published initial pose for AMCL")

    def send_nav_goal(self, pose: PoseStamped):
        if not self._nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("send_nav_goal: action server not available")
            return None
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        send_goal_future = self._nav_action_client.send_goal_async(goal_msg)
        return send_goal_future


    def waitUntilNav2Active(self, timeout_sec=10.0):
        if not self._nav_action_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("waitUntilNav2Active: action server not available")
            return False
        return True
