#!/usr/bin/env python

from datetime import datetime
from PIL import Image
import rclpy
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from rclpy.node import Node

#Initialization HOG Detector
body_hog = cv2.HOGDescriptor()
body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Initialization Multi Tracker 
multiTracker = cv2.legacy.MultiTracker_create()

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
        self.sec = datetime.now().second

    def camera_callback(self, frame):

        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(frame, desired_encoding="bgr8")

            (is_track, self.sec) = track(cv_image, self.sec)
            
            if is_track:
            	msg = Bool()
            	msg.data = True
            	self.publisher_.publish(msg)
            	self.get_logger().info(str(msg.data))

        except CvBridgeError as e:
            print(e)

#Tracker
def track(frame, sec):
    
    success, boxes = multiTracker.update(frame)

    #Update data once per second
    now = datetime.now().second
    check = True if (now - sec == 1) or (sec - now == 59) else False
    sec = now

    if check:
        bodies = find_object(frame)

        #Add new people
        for (x, y, w, h) in bodies:
            hit = True
            x0, y0 = (x + w/2, y + h/2)

            for i, newbox in enumerate(boxes):
                if x0 > newbox[0] and y0 > newbox[1] and x0 < newbox[0] + newbox[2] and y0 < newbox[1] + newbox[3]:  
                    hit = False
                    break
                
            if hit:
                tracker = cv2.legacy.TrackerKCF_create()
                multiTracker.add(tracker, frame, (x, y, w, h))
    
    count = 0   
    for i, newbox in enumerate(boxes):
        if newbox[2] == 0 and newbox[3] == 0:
            count += 1

    ans = False if (count != 0 and count == len(boxes)) or len(boxes) == 0 else True
    return (ans, sec)



#Object detection
def find_object(frame):
    
    frame = dsize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))

    return bodies

#Resize
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
