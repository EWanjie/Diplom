#!/usr/bin/env python

from datetime import datetime
import rclpy
import dlib
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from rclpy.node import Node

#Initialization HOG Body Detector
body_hog = cv2.HOGDescriptor()
body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Initialization HOG Face Detector
face_hog = dlib.get_frontal_face_detector()

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
    #new_people = False
    now = datetime.now().second
    check = True if (now - sec == 1) or (sec - now == 59) else False
    sec = now

    if check:
        bodies = find_object(frame)

        #Add new people
        for body in bodies:
            if in_rect(body, boxes):
                tracker = cv2.legacy.TrackerKCF_create()
                multiTracker.add(tracker, frame, body)
                
    count = 0   
    for i, newbox in enumerate(boxes):
        if newbox[2] == 0 and newbox[3] == 0:
            count += 1

    ans = False if (count != 0 and count == len(boxes)) or len(boxes) == 0 else True
    return (ans, sec)


#Object detection
def find_object(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))
    people = [(x, y, w, h) for x, y, w, h in bodies]

    rects = face_hog(gray)
    faces = [convert(frame, r) for r in rects]

    for face in faces:
        if in_rect(face, bodies):
            people.append(face)
            
    return people

#Convert coordinates
def convert(image, rect):
    start_x = max(0, rect.left())
    start_y = max(0, rect.top())
    end_x = min(rect.right(), image.shape[1])
    end_y = min(rect.bottom(), image.shape[0])

    w = end_x - start_x
    h = end_y - start_y

    return start_x, start_y, w, h

#This is a new objetc?
def in_rect(point, rectangles):

    hit = True
    x0, y0 = (point[0] + point[2]/2, point[1] + point[3]/2)

    for (x2, y2, w2, h2) in rectangles:
            if x0 > x2 and y0 > y2 and x0 < x2 + w2 and y0 < y2 + h2:  
                hit = False
                break     
    return hit


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
