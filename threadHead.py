import cv2
import os
import numpy as np
import threading
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class ClientThread(threading.Thread):


    def __init__(self, img, interval, framerate):
        self.img = img
        threading.Thread.__init__(self)
        self.facePos = []
        self.interval = interval
        self.timer0 = 0
        self.stop = False
        self.framerate = framerate

    def run(self):
        print("thread launched")
        try:
            i = 100
            while True:

                """
                                if i > self.framerate*self.interval:
                    
                    self.facePos = face_cascade.detectMultiScale(cv2.cvtColor(self.cap.read(), cv2.COLOR_BGR2GRAY), 1.1, 10)
                    i = 0
                else:
                    self.allFrames.append(self.cap.read())
                    i+=1
                """


                if time.time() - self.timer0 > self.interval:
                    self.facePos = face_cascade.detectMultiScale(self.img, 1.1, 10)
                    #print("detect faces")
                    self.timer0 = time.time()
                else:
                    time.sleep(self.interval/10)

        except Exception as e:
            print(e)
            print("stop detect")
            pass

    def applyMask(self):
        return 0