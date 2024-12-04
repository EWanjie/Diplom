# pip install dlib
# pip install Pillow
# pip install opencv-python
# pip install opencv-contrib-python

from PIL import Image
import numpy as np
import pickle
import dlib
import cv2
import os


def body_detection():

    cap = cv2.VideoCapture(0)

    body_hog = cv2.HOGDescriptor()
    body_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    face_hog = dlib.get_frontal_face_detector()

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            boxes, weights = body_hog.detectMultiScale(gray, winStride=(8, 8))

            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            rects = face_hog(gray)
            faces = [convert(frame, r) for r in rects]

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


            cv2.imshow("frame", frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def add_face():

    cap = cv2.VideoCapture(0)
    face_hog = dlib.get_frontal_face_detector()

    try:
        names = pickle.load(open('names.pickle', 'rb'))
    except EOFError:
        names = []

    name = input('Enter name for the Face: ')
    names.append(name)
    id = names.index(name)

    print('''\n Look in the camera!
        Try to move your face and change expression for better face memory registration.\n''')

    count = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = face_hog(gray)
            faces = [convert(frame, r) for r in rects]

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1

                cv2.imwrite("face_dataset/" + name + "." + str(id) + '.' + str(count) + ".jpg", gray[y: y + h, x: x + w])
                cv2.imshow('frame', frame)

        if cv2.waitKey(100) & 0xFF == 27:
            break
        elif count >= 50:
            break

    with open('names.pickle', 'wb') as f:
        pickle.dump(names, f)

    cap.release()
    cv2.destroyAllWindows()

    face_learning()


def face_learning():
    path = 'face_dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_hog = dlib.get_frontal_face_detector()

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    known_encodings = []
    known_names = []

    for imagePath in image_paths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        rects = face_hog(img_numpy)
        faces = [convert(img_numpy, r) for r in rects]

        for (x, y, w, h) in faces:
            known_encodings.append(img_numpy[y:y + h, x:x + w])
            known_names.append(id)

    recognizer.train(known_encodings, np.array(known_names))
    recognizer.write('trainer.yml')


def convert(image, rect):
    start_x = max(0, rect.left())
    start_y = max(0, rect.top())
    end_x = min(rect.right(), image.shape[1])
    end_y = min(rect.bottom(), image.shape[0])

    w = end_x - start_x
    h = end_y - start_y

    return start_x, start_y, w, h


def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    face_hog = dlib.get_frontal_face_detector()

    with open('names.pickle', 'rb') as f:
        names = pickle.load(f)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = face_hog(gray)
            faces = [convert(frame, r) for r in rects]

        for(x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if confidence < 100:
                id = names[id]
            else:
                id = "unknown"

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    body_detection()
    #add_face()
    #face_recognition()