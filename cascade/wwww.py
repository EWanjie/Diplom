import cv2
import numpy as np
from PIL import Image
import pickle
import time
import dlib
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



def faceSampling():
    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    name = input('Enter name for the Face: ')

    try:
        names = pickle.load(open('names.pickle','rb'))
    except EOFError:
        names =[]

    names.append(name)
    id = names.index(name)




    print('''\n
    Look in the camera Face Sampling has started!.
    Try to move your face and change expression for better face memory registration.\n
    ''')
    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset2/"+name+"." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 80: # Take 80 face sample and stop video
             break

    with open('names.pickle', 'wb') as f:
        pickle.dump(names, f)

    # Do a bit of cleanup
    print("Your Face has been registered as {}\n\nExiting Sampling Program".format(name.upper()))
    cam.release()
    cv2.destroyAllWindows()

def faceLearning():
    # Path for face image database
    path = 'dataset2'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer.yml')

    # Print the numer of faces trained and end program
    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids))))


def faceRecognition():
    total = 0
    frame_count = 0

    print('\nStarting Recognizer....')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    face_hog = dlib.get_frontal_face_detector()

    #cascadePath = "haarcascade_frontalface_default.xml"
    #faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Starting realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    with open('names.pickle', 'rb') as f:
        names = pickle.load(f)

    while True:

        ret, img =cam.read()

        start = time.time()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #faces = faceCascade.detectMultiScale(
        #    gray,
        #    scaleFactor = 1.2,
        #    minNeighbors = 5,
        #    minSize = (int(minW), int(minH)),
        #   )

        rects = face_hog(gray)
        faces = [convert_and_trim_bb(img, r) for r in rects]

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
            else:
                id = "unknown"

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)

            end = time.time()
            fps = 1 / (end - start)
            total += fps
            frame_count += 1

        cv2.imshow('camera',img)

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    avg_fps = total / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    # Do a bit of cleanup
    print("\nExiting Recognizer.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #faceSampling()
    #faceLearning()
    faceRecognition()