import mediapipe as mp
import cv2 as cv
import numpy as np
import time 

# Importing necessary tools 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# using camera to capture the video frame 
cap = cv.VideoCapture(0)

while True: 
    success, frame = cap.read()
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                if id == 11: 
                    cv.circle(frame, (cx,cy), 12, (0,0,255), thickness=1)

            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
    cv.imshow("Image", frame)
    cv.waitKey(1)



