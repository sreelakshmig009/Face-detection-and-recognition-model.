
import cv2
import numpy as np
from PIL import Image
import os


path = "E:\Face detection and recognition model\image" #I encountered an error in the path so I shifted the image captured from my admin users folder to another folder called image which is located in the same directory as the dataset folder before training it

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("E:\Face detection and recognition model\dataset\haarcascade_frontalface_default.xml")

# function to get the images and label data(originally this program was written to train number of images but I modified it according to my needs)
def getImagesAndLabels(path):

        
    faceSamples=[]
    ids = []

    

    PIL_img = Image.open(path).convert('L') 
    img_numpy = np.array(PIL_img,'uint8')

    id = int(os.path.split(path)[-1].split(".")[1])
    faces = detector.detectMultiScale(img_numpy)

    for (x,y,w,h) in faces:

        faceSamples.append(img_numpy[y:y+h,x:x+w])
        ids.append(id)

    return faceSamples,ids

print ("\nTraining faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print the number of faces trained
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
