import cv2
from deepface import DeepFace
import numpy as np
import os
import time
import threading
import threadHead
import threadEmotion
import tensorflow as tf

def create_face_dic():

    #format =>  'grinning_face': [left eye, rig]

    dict = {}
    list = []
    dire = 'emojis/'
    emojisName = os.listdir(dire)
    print(emojisName)
    for name in emojisName:
        path = dire + name
        img = cv2.imread(path)
        dict[name[:-4]] = img
        emojiList.append(img)



def get_emoji_mood(roi_color,roi_gray):

    eyes = eye_cascade.detectMultiScale(roi_gray)
    #print(eyes)
    eyes_pos = []
    eyes_open = [1,1]  #[l,r]
    incl = 0

    eyes = sorted(eyes,key=lambda eyes: eyes[1])
    for (ex, ey, ew, eh) in eyes[:2]:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eyes_pos.append((int(ex + ew/2), int(ey + eh/2)))


        cv2.rectangle(roi_color, (int(ex + ew/2), int(ey + eh/2)), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.imshow('img',roi_color)
        cv2.waitKey(0)

    if len(eyes_pos) == 2:
        deltaY = eyes_pos[1][1]-eyes_pos[0][1]
        deltaX = eyes_pos[1][0]-eyes_pos[0][0]

        incl = -np.arctan(deltaY/deltaX)
    return emojiList[2], incl

new_model= tf.keras.models.load_model('models/my_model_64p35.h5')

emotion_list = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprised']

def get_emotion(roi_gray):
    final_image = cv2.resize(roi_gray,(224,224))

    final_image = np.expand_dims(final_image,axis=0)

    final_image=final_image/255.0

    Prediction = new_model.predict(final_image)
    sum=0
    for i in Prediction[0]:
        sum+=i
    prediction_scaled=[]
    for i in Prediction[0]:
        prediction_scaled.append(i/sum)
    print(prediction_scaled)
    print(emotion_list[np.argmax(Prediction)])

def apply_emoji(roi_color,emoji, incl):

    emoji_sized = cv2.resize(emoji, roi_color.shape[:-1])
    img2gray = cv2.cvtColor(emoji_sized, cv2.COLOR_BGR2GRAY)

    rot = cv2.getRotationMatrix2D((img2gray.shape[0]/2,img2gray.shape[1]/2),(incl*180)/np.pi,1)

    img2gray = cv2.warpAffine(img2gray, rot, (img2gray.shape[0], img2gray.shape[1]))
    emoji_sized = cv2.warpAffine(emoji_sized, rot, (img2gray.shape[0], img2gray.shape[1]))


    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)


    img1_bg = cv2.bitwise_and(roi_color, roi_color,mask = mask_inv)
    emoji_fg = cv2.bitwise_and(emoji_sized, emoji_sized , mask=mask)

    dst = cv2.add(img1_bg, emoji_fg)

    return dst


emojiList = []
create_face_dic()

###################
#CONSTANTS        #
###################

faceDetectInterval = 2
timer0 = time.time() #face detection timer

emotionDetectInterval = 0.5
timer1 = time.time() #emotion detection timer
timerLoop = time.time()
framerate =3
print_lock = threading.Lock()

# Load the cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture("sampleVideo/group0.mp4")
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
_, img = cap.read()
a = np.zeros(shape=img.shape, dtype=np.int8)



i=0


if __name__ =="__main__":
    threadFace = threadHead.ClientThread(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), faceDetectInterval, cap, framerate)
    threadFace.start()


    while True:

        try:

            if time.time() - timerLoop > 1/framerate:

                timerLoop = time.time()
                # Read the frame
                _, img = cap.read()

                #result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect the faces


                # Draw the rectangle around each face

                if time.time() - timer0 > faceDetectInterval:
                    faces = threadFace.facePos
                    threadFace.img = gray
                    timer0 = time.time()

                """
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,
                            result['dominant_emotion'],
                            (50, 50),
                            font, 3,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)    
                """
                faces = threadFace.facePos
                #
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    roi_color = img[y:y + h, x:x + w]
                    print((x, y, x + w, y + h))
                if len(faces)==0:
                    roi_color=img
                if time.time() - timer1 > emotionDetectInterval:
                    #get_emotion(roi_color)
                    timer1 = time.time()

                # Display
                #dst = cv2.add(img, a)
                cv2.imshow('img', img)
                #print(i)
                i+=1


            # Stop if q key is pressed
            if cv2.waitKey(2) & 0xff == ord('q'):
                break
        except Exception as e:
            print("stop")
            print(e)
            #threadFace.stop = True
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindow()
