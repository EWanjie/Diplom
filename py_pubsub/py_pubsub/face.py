#!/usr/bin/env python

import rclpy
import dlib
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.node import Node


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_sub_face')
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
            
            face_hog = dlib.get_frontal_face_detector()
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            rects = face_hog(gray)
            
            if len(rects):
            	print('find')
            
            
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
