#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import copy

class PointCloudSamplerForNav2(Node):
    def __init__(self):
        super().__init__('pointcloud_sampler_for_nav2')
        # 파라미터 선언
        self.declare_parameter('input_topic', '/oakd/rgb/preview/depth/points')
        self.declare_parameter('output_topic', '/oakd/rgb/preview/depth/points_sampled_for_nav2')
        self.declare_parameter('output_hz', 2.0)
        self.declare_parameter('stride', 10)
        self.declare_parameter('min_z', 0.0)
        self.declare_parameter('max_z', 2.0)
        self.declare_parameter('max_distance', 3.0)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_hz = self.get_parameter('output_hz').value
        self.stride = self.get_parameter('stride').value
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value
        self.max_distance = self.get_parameter('max_distance').value

        self.last_msg = None
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cloud_callback, 10)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)

        if self.output_hz > 0:
            timer_period = 1.0 / self.output_hz
            self.timer = self.create_timer(timer_period, self.timer_publish)

        self.get_logger().info(
            f"PointCloud sampler for Nav2: '{self.input_topic}' -> '{self.output_topic}', rate={self.output_hz}Hz, stride={self.stride}"
        )

    def cloud_callback(self, msg: PointCloud2):
        # 최신 메시지 저장
        self.last_msg = msg

    def timer_publish(self):
        if self.last_msg is None:
            return
        msg = self.last_msg

        sampled_pts = []
        for idx, p in enumerate(point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True)):
            if idx % self.stride != 0:
                continue
            x, y, z = p[0], p[1], p[2]
            if z < self.min_z or z > self.max_z:
                continue
            if x*x + y*y + z*z > (self.max_distance ** 2):
                continue
            sampled_pts.append([x, y, z])
        header = copy.deepcopy(msg.header)
        header.stamp = self.get_clock().now().to_msg()
        new_msg = point_cloud2.create_cloud_xyz32(header, sampled_pts)
        self.pub.publish(new_msg)
        self.get_logger().debug(f"Published sampled PointCloud2 for Nav2: {len(sampled_pts)} pts")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSamplerForNav2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
