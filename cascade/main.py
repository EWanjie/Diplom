# pip install dlib
# pip install face_recognition
# pip install imutils
# pip install opencv-python

# from yoloface import face_analysis
from imutils import paths
from PIL import Image
import face_recognition
import numpy as np
import pickle
import dlib
import time
import sys
import cv2
import os

def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    w = endX - startX
    h = endY - startY

    return (startX, startY, w, h)


def train_model():
    if not os.path.exists("dataset"):
        sys.exit()

    imagePaths = list(paths.list_images("dataset"))

    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):

        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")   # model="cnn"
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    data = {
        "encodings": knownEncodings,
        "name": knownNames
    }

    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def face_distinction():

    data = pickle.loads(open("encodings.pickle", "rb").read())
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if ret:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            names = []
            
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = data["name"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(img, (left, top), (right, bottom),
                              (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

        cv2.imshow("img", img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def move_detector():

    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=3)

        сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in сontours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("frame1", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def HOG_detection():

    # Source ##############################################################

    #cap = cv2.VideoCapture('video.mp4')
    cap = cv2.VideoCapture(0)

    #######################################################################



    # Haar ################################################################

    body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #######################################################################



    # HOG #################################################################

    body_hog = cv2.HOGDescriptor()
    body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    face_hog = dlib.get_frontal_face_detector()

    #######################################################################



    # YOLO ################################################################

    # face = face_analysis()

    #######################################################################



    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15., (640, 360))
    total = 0
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            start = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # boxes = body_cascade.detectMultiScale(gray, 1.3, 5)
            #bboxes = face_cascade.detectMultiScale(gray, 1.3, 5)

            # boxes, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))

            rects = face_hog(gray)
            boxes = [convert_and_trim_bb(frame, r) for r in rects]


            # __, boxes, conf = face.face_detection(frame_arr=frame,frame_status=True, model='full')
            # frame = face.show_output(img=frame, face_box=boxes, frame_status=True)

            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            end = time.time()
            fps = 1 / (end - start)
            total += fps
            frame_count += 1

            cv2.imshow("frame", frame)
            out.write(frame)

            wait_time = max(1, int(fps/4))

            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break
        else:
            break

    avg_fps = total / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #train_model()                             # Create face database
    #face_distinction()                        # Face detection and recognition // Somesimg not LTS
    #move_detector()                           # Motion Detection
    HOG_detection()


