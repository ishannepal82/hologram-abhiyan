import mediapipe as mp
import cv2 as cv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv.VideoCapture(0)

while True: 
    success, frame = cap.read()
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(w*lm.x), int(h*lm.y)
            
        
    cv.imshow("Image", frame)
    cv.waitKey(1)
        
    