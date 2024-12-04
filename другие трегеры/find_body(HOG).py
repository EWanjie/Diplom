#!/usr/bin/env python

import rclpy
import cv2

from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from rclpy.node import Node

body_hog = cv2.HOGDescriptor()
body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('find_body')
        
        self.number = 10
        
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.camera_callback,
            self.number)
        
        self.publisher_ = self.create_publisher(Bool, 'is_body', self.number)
                    
        self.bridge_object = CvBridge()

    def camera_callback(self, frame):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(frame, desired_encoding="bgr8")
            
            cv_image = dsize(cv_image)

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            boxes, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))

            if len(boxes):
            	msg = Bool()
            	msg.data = True
            	self.publisher_.publish(msg)
            	self.get_logger().info(str(msg.data))

        except CvBridgeError as e:
            print(e)



def dsize(frame):
    width = 500
    height = int(frame.shape[0] * (width / frame.shape[1]))
    dsize = (width, height)
    frame = cv2.resize(frame, dsize)
    return frame


def main(args=None):

    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    rclpy.spin(camera_subscriber)

    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
