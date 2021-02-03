import cv2
from deepface import DeepFace

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
def create_face_dic():
    print("ok")

create_face_dic()



while True:
    # Read the frame
    _, img = cap.read()
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                result['dominant_emotion'],
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
    # Display
    cv2.imshow('img', img)

    # Stop if q key is pressed
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindow()
