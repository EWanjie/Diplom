#!/usr/bin/env python

import rclpy
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.node import Node


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.camera_callback,
            10)

        self.i = 0
        self.bridge_object = CvBridge()

    def camera_callback(self, msg):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            body_hog = cv2.HOGDescriptor()
            body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            boxes, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))

            for (x, y, w, h) in boxes:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            self.i += 1

            cv2.imwrite('/home/dreamer/ros2_ws/src/py_pubsub/py_pubsub/1/%s.png' % self.i, cv_image)

            if len(boxes):
                self.get_logger().info('Find')

        except CvBridgeError as e:
            print(e)


def main(args=None):
    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    rclpy.spin(camera_subscriber)

    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
