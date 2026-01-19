import mediapipe as mp
import cv2 as cv
import numpy as np
import time 
import math

class PoseDetector ():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        

    def detect_pose(self, image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image
    
    def detect_postion(self, image, hand_no=0, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
           for id, lm in enumerate(self.results.pose_landmarks.landmark):
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
        
    def detecct_hand_raise (self, lm_list):
        right_shoulder = lm_list[12][0]
        right_wrist = lm_list[16][0]
        
        left_shoulder = lm_list[11][0]
        left_wrist = lm_list[15][0]
         
        if right_wrist['y'] < right_shoulder['y'] or left_wrist['y'] < left_shoulder['y']:
            return True
            
        return False
    
    def detect_hands_crossed (self, lm_list):
        right_wrist = lm_list[16][0]
        right_elbow = lm_list[14][0]
        left_wrist = lm_list[15][0]
        left_elbow = lm_list[13][0]
        
        distance_right = math.sqrt(
            (right_elbow['x'] - left_wrist['x']) ** 2 +
            (right_elbow['y'] - left_wrist['y']) ** 2
        )
        
        distance_left = math.sqrt(
            (left_elbow['x'] - right_wrist['x']) ** 2 +
            (left_elbow['y'] - right_wrist['y']) ** 2
        )
        
        if distance_left < 20 and distance_right < 20:
            return True
        return False
        
    # def detect_okay_hand_sign (self, lm_list):
    #     thumb = lm_list[4][0]
    #     index = lm_list[8][0]

    #     distance = math.sqrt(
    #         (thumb['x'] - index['x']) ** 2 +
    #         (thumb['y'] - index['y']) ** 2
    #     )
        
        
    #     if distance < 20:  # finger extended
    #         middle_tip = lm_list[12][0]
    #         middle_pip = lm_list[10][0]
    #         if middle_tip['y'] < middle_pip['y']:
    #             return True
            
    #     return False

def main():
    cap = cv.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, frame = cap.read()
        frame = detector.detect_pose(frame)
        lm_list = detector.detect_postion(frame, draw=True)
        if len(lm_list) != 0:
            if detector.detect_hands_crossed(lm_list):
                cv.putText(frame, "HandCrossed", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv.imshow("Image", frame) 
        cv.waitKey(1)

if __name__ == "__main__":
    main()