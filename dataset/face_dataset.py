from cv2 import cv2
import os

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW) #0 to turn on the webcam and cv2.CAP_DSHOW to indicate that the webcam is turned on and ready for image capture
cam.set(3,640) #set image width
cam.set(4,480) #set image height

face_detector = cv2.CascadeClassifier("E:\Face detection and recognition model\dataset\haarcascade_frontalface_default.xml") #path where the haarcascade xml file is located.Note that I have copy-pasted the full path of the file and replaced front slashes with backslash in the path

face_id = input("\n Enter user-id") # For each person in the image capture, enter one numeric face id

print("\n Initializing face capture....")

count = 0

while(True):

    ret, img = cam.read()

    if ret == False:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5) #detectMultiScale detect objects(here it is person) of different sizes in the input image

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #captured image will be in the form of a rectangle in the predefined set of dimensions
        count += 1
        
        cv2.imshow('image', gray) #displays the window with image to be captured

    k = cv2.waitKey(1) & 0xff #waitKey waits for the user to input a specified key to terminated or continue a specified action
    
    if k == ord('s') :
        cv2.imwrite('image.jpg',gray) #writes the image into the image.jpg file(here I'm training only one image. If you want, you can train any number of images )
        cam.release()
        cv2.destroyAllWindows()
        

print("\nExiting the program")


  


