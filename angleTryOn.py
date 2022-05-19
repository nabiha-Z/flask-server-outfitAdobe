
import json
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import imutils
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('TestingVideo7.mp4') #0 for irium 1 for webcam
cap.set(3,1100)   #width
cap.set(4,900)    #height
cap.set(10,200)
takeSS = False
start_time = time.time()
capture_duration = 9
landmarks = ""
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(L,R,h,w):
    # print("left shoulder x-axis: ", L[0]*w)
    # print("left shoulder y-axis: ", L[1]*h)
    s2=L[0]
    t2=L[1]
    s1=R[0]
    t1=R[1]
    shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
    shoulderslength = round(shoulderslength)
    return shoulderslength

def angletryon():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while (int(time.time() - start_time) < capture_duration):
            success, image = cap.read()
            #1 means horizontal 
            image = cv2.flip(image,1)
            h, w, c = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # # Make detection
            results = pose.process(image)
            
            # # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not success:
                print("Ignoring empty camera frame.")
                continue
            imgae = cv2.putText(image, "TryOn Angle Screen", (400,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                # Calculate angle
                length = calculate_angle(shoulderL,shoulderR,h,w)
                X=shoulderL[0]*w
                Y=shoulderL[1]*h
                center_coordinates=(X,Y)
                axesLength = (100, 50)
                angle = 0
                startAngle = 0
                endAngle = 360
                color = (255, 0, 0)
                data="H"
                LXP1=(shoulder[0])-0.05
                LYP2=(shoulder[1])
                LS1=LXP1,LYP2
                LXP3=(shoulder[0])+0.05
                LYP4=(shoulder[1])
                LS2=LXP3,LYP4
               

                cv2.putText(image, str(data), 
                            tuple(np.multiply(shoulder, [320, 610]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 247, 237), 2, cv2.LINE_AA
                                    )
                
                                    
                # Line thickness of -1 px
                thickness = -1
                if(length<7):
                    cv2.putText(image, str(length), 
                            tuple(np.multiply(elbow, [240, 280]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 247, 237), 2, cv2.LINE_AA
                                    )     
                    cv2.putText(image, str(data), 
                            tuple(np.multiply(LS1, [320, 610]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 29, 174), 2, cv2.LINE_AA
                                    )
                    cv2.putText(image, str(data), 
                            tuple(np.multiply(LS2, [320, 610]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 23, 23 ), 2, cv2.LINE_AA
                                    ) 
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                       ) 
            cv2.imshow('Measurements', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if len(landmarks)!=0:
            s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            s1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            t1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
            shoulderslength = round(shoulderslength)
            # shoulderslength1 = int(abs(landmarks[12][1]+landmarks[11][1])/2)
            print("shoulders: ", shoulderslength)

angletryon()