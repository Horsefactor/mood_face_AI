import cv2
import os
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Read the input image
img = cv2.imread('test.jpg')


# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

emojiList = []
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
        dict[name[:-4]] = 0
        emojiList.append(img)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    dict['grinning_face'] = []
    # for img in emojiList:
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)

def get_emoji_mood(roi_color,roi_gray):

    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyes_pos = []
    for (ex, ey, ew, eh) in eyes[:2]:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eyes_pos.append((int(ex + ew/2), int(ey + eh/2)))


        cv2.rectangle(roi_color, (int(ex + ew/2), int(ey + eh/2)), (ex + ew, ey + eh), (0, 255, 0), 2)

    deltaY = eyes_pos[1][1]-eyes_pos[0][1]
    print(deltaY)

    deltaX = eyes_pos[1][0]-eyes_pos[0][0]
    print(deltaX)

    incl = np.arctan(deltaY/deltaX)

    print(incl)
    print(eyes_pos)
    print("---")
    return emojiList[2] , incl

def apply_emoji(roi_color,emoji, incl):

    emoji_sized = cv2.resize(emoji, roi_color.shape[:-1])

    img2gray = cv2.cvtColor(emoji_sized, cv2.COLOR_BGR2GRAY)


    ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi_color, roi_color,mask = mask_inv)



    emoji_fg = cv2.bitwise_and(emoji_sized, emoji_sized , mask=mask)
    # rot = cv2.getRotationMatrix2D((emoji_fg.shape[0]/2,emoji_fg.shape[1]/2),90,1)
    # print(rot)
    # emoji_fg = cv2.warpAffine(img, rot, (emoji_fg.shape[0], emoji_fg.shape[1]))

    dst = cv2.add(img1_bg, emoji_fg)

    #dst = cv2.addWeighted(img1_bg,0.5,emoji_fg,0.5,0)
    return dst


create_face_dic()

for (x, y, w, h) in faces:

    #img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    emoji, incl= get_emoji_mood(roi_color,roi_gray)

    dst = apply_emoji(roi_color,emoji, incl)
    img[y:y + h, x:x + w] = dst
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Display the output
emo = emojiList[0]
# img = cv2.addWeighted(img,0.7,emo,0.3,0)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw rectangle around the faces
