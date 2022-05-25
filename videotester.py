from hashlib import new
import os
from turtle import onclick
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import webbrowser

# load model
# webbrowser.open('https://www.google.com',new=new)
model = load_model("best_model.h5")
playlist={'angry':'https://open.spotify.com/playlist/0KPEhXA3O9jHFtpd1Ix5OB','disgust':'https://open.spotify.com/playlist/3qgzMg4m5tvf16PzlPgGa9','fear':'https://open.spotify.com/playlist/7rzS9iLiqjy65AsZd9qinf','happy':'https://open.spotify.com/playlist/1llkez7kiZtBeOw5UjFlJq','sad':'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1','surprise':'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0','neutral':'https://open.spotify.com/track/53ISyyRA6cCY3U4pL9CSYG'}

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



cap = cv2.VideoCapture(0)
global captured,captured_emotion
captured=0
captured_emotion=""

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    predicted_emotion=""

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('c'):  # wait until 'c' key is pressed
        captured=1
        captured_emotion=predicted_emotion
        break

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows

def show_playlist():
    global captured,captured_emotion
    window=Tk()
    window.geometry('500x500')
    window.resizable(0,0)
    window.title('PlayList for your mood')
    window.configure(bg="cyan")
    Label(window,text="Your mood now is "+captured_emotion, font="rockwell 15 bold").place(x=10,y=10)
    url=playlist.get(captured_emotion)
    def openSpotify(url):
        webbrowser.open(url,new=new)
    Button(window,text="Open Spotify",command=openSpotify(url)).place(x=100,y=50)
    window.mainloop()
if(captured==1):
    show_playlist()

