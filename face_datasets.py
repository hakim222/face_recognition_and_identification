import os
import cv2

cam = cv2.VideoCapture(0)

# set video width and height
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# for each new face, enter unique id
face_id = input('\n Type user id and press ENTER -> ')
print('\n Initializing face registration. Look the camera and wait...')

count = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # for every faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count +=1
        # save captured image in datasets folder with proper name
        cv2.imwrite("datasets/User." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
    
    # exit the program if completed capturing 30 img or pressing the ESC key
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        print('\n Image capture completed.')
        break

cam.release()
cv2.destroyAllWindows()

