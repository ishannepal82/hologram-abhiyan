import mediapipe as mp
import cv2 as cv
import numpy as np
import time 
import math

class HandTracker ():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        

    def detect_hands(self, image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image, handLms, self.mp_hands.HAND_CONNECTIONS)
        return image
    
    def detect_postion(self, image, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
           my_hand = self.results.multi_hand_landmarks[hand_no]
           for id, lm in enumerate(my_hand.landmark):
             h, w, c = image.shape
             cx, cy = int(w*lm.x), int(h*lm.y)
             cz = lm.z  # keep as float (depth info)
             lm_list.append([{
                 'id': id,
                 'x': cx,
                 'y': cy,
                 'z': cz
             }])
             if draw:
                cv.circle(image, (cx,cy), 12, (0,0,255), thickness=1)
        return lm_list
        
    def detect_okay_hand_sign (self, lm_list):
        thumb = lm_list[4][0]
        index = lm_list[8][0]

        distance = math.sqrt(
            (thumb['x'] - index['x']) ** 2 +
            (thumb['y'] - index['y']) ** 2
        )
        
        
        if distance < 20:  # finger extended
            middle_tip = lm_list[12][0]
            middle_pip = lm_list[10][0]
            if middle_tip['y'] < middle_pip['y']:
                return True
            
        return False

def main():
    cap = cv.VideoCapture(0)
    detector = HandTracker()
    while True:
        success, frame = cap.read()
        frame = detector.detect_hands(frame)
        lm_list = detector.detect_postion(frame, draw=False)
        if len(lm_list) != 0:
           if detector.detect_okay_hand_sign(lm_list):
                cv.putText(frame, "OK Sign Detected", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv.imshow("Image", frame) 
        cv.waitKey(1)

if __name__ == "__main__":
    main()