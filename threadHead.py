import cv2
import os
import numpy as np
import threading
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class ClientThread(threading.Thread):


    def __init__(self, img, interval):
        self.img = img
        threading.Thread.__init__(self)
        self.facePos = []
        self.interval = interval
        self.timer0 = 0


    def run(self):
        print("thread launched")

        while True:
            print("detect faces")
            self.facePos = face_cascade.detectMultiScale(self.img, 1.1, 10)


            time.sleep(self.interval)
        pass

    def applyMask(self):
        return 0