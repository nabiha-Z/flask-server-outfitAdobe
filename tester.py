import requests
import cv2, time
import imutils
import math
import numpy as np
import mediapipe as mp

url ="http://192.168.100.26:8080/shot.jpg"

mp_pose = mp.solutions.pose
landmarks = ""
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False     
        # Make detection
        results = pose.process(img)     
        # Recolor back to BGR
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, 'Measruements',(300, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (305, 96, 202), 3, cv2.LINE_AA) 
       
        # # Extract landmarks
        try:
             landmarks = results.pose_world_landmarks.landmark 
             print("detected")
             s2 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x]
             t2 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
             s1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x]
             t1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
             print("s1: ", s1)
             shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
             print("shoulders: ", shoulderslength)
             shoulderslength = round(shoulderslength)
             # shoulderslength1 = int(abs(landmarks[12][1]+landmarks[11][1])/2)
             print("shoulders: ", shoulderslength)
        except:
            print("pass")
            pass
                
        # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        #                             ) 
                                        
        cv2.imshow("Android_cam", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

print("ghfjhj")
s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
s1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
t1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
shoulderslength = round(shoulderslength)
# shoulderslength1 = int(abs(landmarks[12][1]+landmarks[11][1])/2)
print("shoulders: ", shoulderslength)