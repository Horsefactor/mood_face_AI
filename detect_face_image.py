import cv2
import os
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


create_face_dic()

for (x, y, w, h) in faces:
    emoji = emojiList[0]
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    img2gray = cv2.cvtColor(emoji, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)


    img1_bg = cv2.bitwise_and(roi_color, roi_color, mask=mask_inv)
    emoji_fg = cv2.bitwise_and(emoji, emoji, mask=mask)
    dst = cv2.add(img1_bg, emoji_fg)
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
