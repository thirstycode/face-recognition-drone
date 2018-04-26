import cv2
import os
import numpy as np
import sys
import time
import logger

from config import *

# scalefactorvalue = 1.2

subjects = [""]
status = [""]

def colour_f(status1):
    if status1=="vip":
        return (0,255,0)
    if status1=="blacklisted":
        return (0,0,255)
    else :
        return (255,255,255)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def detect_face2(img):
    global scalefactorvalue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scalefactorvalue, minNeighbors=5);

    if (len(faces) == 0):
        return ()

    return faces

def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []

    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;

            if image_name == "name.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path,'r+') as name:
                    content = name.read()
                    subjects.append(content)

            elif image_name == "status.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path,'r+') as name:
                    content = name.read()
                    status.append(content)

            else :
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)

                # make sure to resize on fixed amount
                # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                print("Faces Scanned: ", len(faces))
                cv2.waitKey(100)

                face, rect = detect_face(image)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# def onTrackChange(trackvalue):
#     if trackvalue >= 0:
#         global scalefactorvalue
#         trackvalue = trackvalue + 102
#         scalefactorvalue = trackvalue/100
#     else :
#         pass

    # print(scalefactorvalue)

# ### Train Face Recognizer

# As we know, OpenCV comes equipped with three face recognizers.

# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`

# I am going to use LBPH face recognizer but you can use any face recognizer of your choice. No matter which of the OpenCV's face recognizer you use the code will remain the same. You just have to change one line, the face recognizer initialization line given below.


#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

def draw_rectangle2(img, rect,colour):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect face from the image
    faces = detect_face2(img)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            #predict the image using our face recognizer
            label, confidence = face_recognizer.predict(gray[y:y+h,x:x+w])
            #get name of respective label returned by face recognizer
            label_text = subjects[label]
            status_1 = status[label]
            if confidence < 50 :
                #draw a rectangle around face detected
                colour_2 = colour_f(status_1)
                draw_rectangle2(img, (x,y,w,h),colour_2)
                #draw name of predicted person
                draw_text(img, label_text + " " + str(int(100 - confidence + 20)) + "%", x, y-5)
                logger.increment(label_text,status_1)

                # print("face detected")
            else:
                #draw a rectangle around face detected
                draw_rectangle2(img, (x,y,w,h),(255,255,255))
                #draw name of predicted person
                draw_text(img, "No Match", x, y-5)
                # print("non identified face")
        return img
    else :
        return img

print("Predicting Faces From Video...")

logger.start(subjects)

vid = cv2.VideoCapture(0)
# vid.open(ip_address)
frame_count = 0


time_started = time.time()
# windowName="Edit This To Change Scale"
# cv2.namedWindow(windowName)
# onTrackChange(120)
# cv2.createTrackbar("Scale",windowName,0,78,onTrackChange)
while True:
    frame_count += 1

    check , test_img1 = vid.read()
    predicted_img1 = predict(test_img1)
    cv2.imshow("face", cv2.resize(predicted_img1, (1000, 562)))
    # cv2.imshow("face",predicted_img1,predicted_img1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
time_ended = time.time()
print("FPS ==> " + str(frame_count/(time_ended - time_started)))
logger.end1()
