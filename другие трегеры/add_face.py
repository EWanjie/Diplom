#!/usr/bin/env python

import numpy as np
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

name = input('Enter name for the Face: ')

if not os.path.exists('face_dataset'):
    os.mkdir('face_dataset')

if not os.path.exists('names.pickle') or not os.path.exists('trainer.yml'):
    print('Problem opening files')

try:
    names = pickle.load(open('names.pickle', 'rb'))
except EOFError:
    names = []
        
names.append(name)
person_id = names.index(name)


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('add_face')

        self.number = 10

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.camera_callback,
            self.number)
        
        self.count = 1
        self.bridge_object = CvBridge()

    def camera_callback(self, frame):

        try:
                if self.count >= 50:
                    raise SystemExit

                cv_image = self.bridge_object.imgmsg_to_cv2(frame, desired_encoding="bgr8")
                cv_image = dsize(cv_image)
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                rects = face_hog(gray)

                faces = [convert(cv_image, r) for r in rects]

                for(x, y, w, h) in faces:
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.imwrite("face_dataset/" + name + "." + str(person_id) + '.' + str(self.count) + ".jpg", gray[y: y + h, x: x + w])
                    self.count += 1

        except CvBridgeError as e:
            print(e)


def face_learning():
    path = 'face_dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    known_encodings = []
    known_names = []

    for imagePath in image_paths:
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_numpy = np.array(gray, 'uint8')        

        person_id = int(os.path.split(imagePath)[-1].split(".")[1])
        rects = face_hog(img_numpy)
        faces = [convert(img_numpy, r) for r in rects]

        known_encodings.append(img_numpy)
        known_names.append(person_id)

    recognizer.train(known_encodings, np.array(known_names))
    recognizer.write('trainer.yml')


#Resize
def dsize(frame):
    width = 500
    height = int(frame.shape[0] * (width / frame.shape[1]))
    dsize = (width, height)
    frame = cv2.resize(frame, dsize)
    return frame


def convert(image, rect):
    start_x = max(0, rect.left())
    start_y = max(0, rect.top())
    end_x = min(rect.right(), image.shape[1])
    end_y = min(rect.bottom(), image.shape[0])

    w = end_x - start_x
    h = end_y - start_y

    return start_x, start_y, w, h


def main(args=None):

    print('''\nLook in the camera!\nTry to move your face and change expression for better face memory registration.\n''')
    
    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    try:
        rclpy.spin(camera_subscriber)
        
    except SystemExit:
        print('''\nDone\nWait a little...\n''')
        with open('names.pickle', 'wb') as f:
            pickle.dump(names, f)
            
        face_learning()


    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
