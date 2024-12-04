#!/usr/bin/env python

from PIL import Image
import pickle
import rclpy
import dlib
import cv2
import os

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.node import Node

face_hog = dlib.get_frontal_face_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create()

files_exist = False

try:
    recognizer.read('trainer.yml')
    with open('names.pickle', 'rb') as f:
        names = pickle.load(f)
    files_exist = True
except:
    print ('Problem opening files')

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('find_face')

        self.number = 10

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.camera_callback,
            self.number)

        self.publisher_ = self.create_publisher(String, 'is_face', self.number)

        self.bridge_object = CvBridge()

    def camera_callback(self, frame):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(frame, desired_encoding="bgr8")
            
            cv_image = dsize(cv_image)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            rects = face_hog(gray)

            faces = [convert(cv_image, r) for r in rects]

            if len(faces):
                
                msg = String()
                
                if files_exist:
                    
                    for(x, y, w, h) in faces:
                        person_id, confidence = recognizer.predict(gray[y: y + h, x: x + w])

                        if confidence < 100:
                            person_id = names[person_id]
                        else:
                            person_id = 'Unknown'
                            
                    msg.data = person_id
                    
                else:
                    msg.data = 'Unknown'
                    
                self.publisher_.publish(msg)
                self.get_logger().info(msg.data)
                

        except CvBridgeError as e:
            print(e)


def convert(image, rect):
    start_x = max(0, rect.left())
    start_y = max(0, rect.top())
    end_x = min(rect.right(), image.shape[1])
    end_y = min(rect.bottom(), image.shape[0])

    w = end_x - start_x
    h = end_y - start_y

    return start_x, start_y, w, h

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
