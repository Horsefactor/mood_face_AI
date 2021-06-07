import cv2
import os
import numpy as np
import threading
import time
import tensorflow as tf

new_model= tf.keras.models.load_model('models/my_model_64p35.h5')
emotion_list = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprised']

class ClientThread(threading.Thread):


    def __init__(self, interval, framerate):
        self.img = []
        self.heads = {}
        threading.Thread.__init__(self)
        self.facePos = []
        self.interval = interval
        self.timer0 = 0
        self.stop = False
        self.framerate = framerate


    def loadheads(self,head,img):
        self.heads[head] = img

    def get_emotions(self):
        for key, value in self.heads.items():
            final_image = cv2.resize(value, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            Prediction = new_model.predict(final_image)
            key.setEmotion(np.argmax(Prediction))
            print(emotion_list[np.argmax(Prediction)])


    def run(self) -> None:
        print("EMOTION")
        pass
        # try:
        # while True:
        #     if time.time() - self.timer0 > self.interval:
        #         for head,img in self.heads.items():
        #             final_image = cv2.resize(img, (224, 224))
        #             cv2.imshow('img', final_image)
        #             cv2.waitKey(0)
        #             final_image = np.expand_dims(final_image, axis=0)
        #
        #             final_image = final_image / 255.0
        #
        #             Prediction = new_model.predict(final_image)
        #
        #     for n in self.img:
        #         print(n[0])
                # pass

        """
                        while False:

                if time.time() - self.timer0 > self.interval:
                    for head in self.img:
                        final_image = cv2.resize(head, (224, 224))
                        # cv2.imshow('img', final_image)
                        # cv2.waitKey(0)
                        final_image = np.expand_dims(final_image, axis=0)

                        final_image = final_image / 255.0

                        Prediction = new_model.predict(final_image)

                        return np.argmax(Prediction)
                    # print("detect faces")
                    self.timer0 = time.time()
                else:
                    time.sleep(self.interval / 10)
            pass
            """

        # except Exception as e:
        #     print(e)
        #     print("stop Emotion")
        #     pass