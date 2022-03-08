import cv2
import mediapipe as mp
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def measurements():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    capture_duration = 20
    landmarks = "null"
    d=0
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    detector = FaceMeshDetector(maxFaces=1)

    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
        while (int(time.time() - start_time) < capture_duration):
            
            ret, frame = cap.read()
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f'{int(capture_duration-(time.time()-start_time))} s',(300, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (305, 96, 202), 3, cv2.LINE_AA) 
            image, faces = detector.findFaceMesh(image, draw=False)
    
            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]
                # Drawing
                # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
                # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
                w, _ = detector.findDistance(pointLeft, pointRight)
                W = 6.3
    
                # # Finding the Focal Length
                # d = 50
                # f = (w*d)/W
                # print(f)

                # Finding distance
                f = 840
                d = (W * f) / w
            
            if d < 100:
                cv2.rectangle(image, (220, 30), (400, 60), (35, 96, 202), 3)
                cv2.putText(image, f'{int(d)} cm',(50, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 96, 202), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(image, (20, 30), (70, 60), (190, 13, 13), 3)
                cv2.putText(frame, f'{int(d)} cm',(50, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (190, 13, 13), 3, cv2.LINE_AA)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_world_landmarks.landmark

                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                
                        
                except:
                    pass
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        ) 
        
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    if landmarks!="null":
        s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        s1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        t1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )+1
        shoulderslength =round(shoulderslength)

        x2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        y2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        x1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        y1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        RightArmlength = (math.hypot(x2 - x1, y2 - y1)) * 39.37

        print("Shoulders Length: ",shoulderslength)
        print("Right Arm Length: ",RightArmlength)
        return shoulderslength
    else:
        return "not detected"
    