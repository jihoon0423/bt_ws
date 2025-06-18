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
        self.declare_parameter('stride', 250)         # 여기서 설정 가능
        self.declare_parameter('min_z', 0.1)
        self.declare_parameter('max_z', 1.5)
        self.declare_parameter('max_distance', 2.0)
        self.declare_parameter('max_pts', 500)       # 샘플링 후 최대 500개 포인트

        # 파라미터 값 읽기
        self.input_topic  = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_hz    = self.get_parameter('output_hz').value
        self.stride       = int(self.get_parameter('stride').value)
        self.min_z        = float(self.get_parameter('min_z').value)
        self.max_z        = float(self.get_parameter('max_z').value)
        self.max_distance = float(self.get_parameter('max_distance').value)
        self.max_pts      = int(self.get_parameter('max_pts').value)

        self.last_msg = None
        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self.cloud_callback, 10)
        self.pub = self.create_publisher(
            PointCloud2, self.output_topic, 10)

        if self.output_hz > 0:
            self.timer = self.create_timer(1.0/self.output_hz, self.timer_publish)

        self.get_logger().info(
            f"Sampler init: topic {self.input_topic} → {self.output_topic}\n"
            f" rate={self.output_hz}Hz, stride={self.stride}, "
            f"z∈[{self.min_z},{self.max_z}], dist≤{self.max_distance}, "
            f"max_pts={self.max_pts}"
        )

    def cloud_callback(self, msg: PointCloud2):
        self.last_msg = msg

    def timer_publish(self):
        if not self.last_msg:
            return

        sampled = []
        for idx, p in enumerate(point_cloud2.read_points(
                self.last_msg, field_names=('x','y','z'), skip_nans=True)):
            # 1) stride 샘플링
            if idx % self.stride != 0:
                continue

            x,y,z = p
            # 2) 높이 필터
            if z < self.min_z or z > self.max_z:
                continue
            # 3) 거리 필터
            if (x*x + y*y + z*z) > (self.max_distance**2):
                continue

            sampled.append([x,y,z])
            # 4) 최대 포인트 개수 제한
            if len(sampled) >= self.max_pts:
                break

        # 메시지 생성 & 퍼블리시
        hdr = copy.deepcopy(self.last_msg.header)
        hdr.stamp = self.get_clock().now().to_msg()
        out = point_cloud2.create_cloud_xyz32(hdr, sampled)
        self.pub.publish(out)
        self.get_logger().debug(f"Published {len(sampled)} pts (stride={self.stride})")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSamplerForNav2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
