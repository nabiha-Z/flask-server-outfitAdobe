import json
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import imutils
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def mobilemeasurements(uploaded_file,user):

    print("uploaded file name: ",uploaded_file.name)
    uploaded_file.save('user_videos/video.mp4')
    print("capture")
    cap = cv2.VideoCapture('user_videos/video.mp4')
    cap.set(3,850)
    cap.set(4,850)
    start_time = time.time()
    capture_duration = 6
    landmarks = ""
    d=0
    detector = FaceMeshDetector(maxFaces=1)
    # detector = HandDetector(detectionCon=0.8, maxHands=1)
    # detector = FaceMeshDetector(maxFaces=1)

    # x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    # y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
        while (int(time.time() - start_time) < capture_duration):
            
            ret, image = cap.read()
            image = imutils.resize(image, width=350)
                
            
            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # # Make detection
            results = pose.process(image)
        
            # # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f'{int(capture_duration-(time.time()-start_time))} s',(300, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (305, 96, 202), 3, cv2.LINE_AA) 
            image, faces = detector.findFaceMesh(image, draw=False)
    
            # print("no face detected")
            try:
                # print("extracting .... ")
                landmarks = results.pose_world_landmarks.landmark

            #         # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            #         # Calculate angle
            #         angle = calculate_angle(shoulder, elbow, wrist)

            #         # Visualize angle
            #         cv2.putText(image, str(angle), 
            #                     tuple(np.multiply(elbow, [640, 480]).astype(int)), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                             )
                
                        
            except:
                print("passsss")
                pass
                
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            cv2.imshow('Measurement', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    
    if len(landmarks)!=0:
        print("elloo")
        s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        s1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        t1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
        shoulderslength = round(shoulderslength)
        # shoulderslength1 = int(abs(landmarks[12][1]+landmarks[11][1])/2)
        print("shoulders: ", shoulderslength)

        x2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        y2 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        x1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        y1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        RightArmlength = ((math.hypot(x2 - x1, y2 - y1)) * 39.37) + 1
        RightArmlength = round(RightArmlength)
        print("right arm: ", RightArmlength)

        s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        s1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
        t1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        fullLength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 ) + 1
        fullLength =round(fullLength)
        print("length: ", fullLength)

        s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        s1 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
        t1 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        kneeLength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 )
        kneeLength =round(kneeLength)
        print("knee: ", kneeLength)

        s2 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        s1 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        t1 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        # waistLength =int(abs(bodylmlist[12][1]+bodylmlist[11][1])/2)
        waistLength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 ) + 5
        waistLength =round(waistLength)
        
        waistLength = waistLength + waistLength
        print("waist: ", waistLength)

        s2 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        t2 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        s1 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        t1 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        # waistLength =int(abs(bodylmlist[12][1]+bodylmlist[11][1])/2)
        bottomLength = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 ) + 4
        bottomLength =round(bottomLength)

        s2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        t2 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        s1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
        t1 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        tshirt = ((math.hypot(s2 - s1, t2 - t1)) * 39.37 ) + 2
        tshirt =round(tshirt)

        bodymeasurement = {
            'shoulders' :shoulderslength,
            'fullLength':fullLength,
            'arms':RightArmlength,
            "knee": kneeLength,
            "tshirt":tshirt,
            'bottom':bottomLength,
            "waist":waistLength,
            'user':user.replace('"','')
        }
        obj = {
            'msg':"true",
            'data':bodymeasurement
        }
        return obj
    else:
        obj = {
            'msg': "false"
        }
        return obj