
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time
import mediapipe as mp

def arTryOn():
    stopAR = False
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    startdistance = None
    neck_point_xaxis = 500
    neck_point_yaxis = 500
    scaling = 0
    start_time = time.time()
    capture_duration = 30
    cap = cv2.VideoCapture('TestingVideo3.mp4') #0 for irium 1 for webcam
    cap.set(3,1300)   #width
    cap.set(4,950)    #height
    cap.set(10,200)
    takeSS = False
    detector = HandDetector(detectionCon=0.8)

    while (int(time.time() - start_time) < capture_duration):
        success, img = cap.read()
        #1 means horizontal 
        img = cv2.flip(img,1)
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img = cv2.putText(img, "TryOn Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        if stopAR == False:
            
            cloth = cv2.imread("dress2.png", cv2.IMREAD_UNCHANGED)
            cloth = cv2.resize(cloth, (180,180))
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img)
            bodylmlist=[]
            try:
                for i_d, landmark in enumerate((results.pose_landmarks.landmark)):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x*w), int(landmark.y*h)
                    bodylmlist.append([id,cx,cy])
            except:
                pass
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #---------------------------------Finding neck point---Scaling----------------------------
            try:
                if(bodylmlist[12] and bodylmlist[11]):
                    if startdistance is None:
                        length_of_shoulders = abs(bodylmlist[12][1]-bodylmlist[11][1])
                        startdistance = length_of_shoulders
                        length_of_shoulders = abs(bodylmlist[12][1]-bodylmlist[11][1])
                        scaling = int((length_of_shoulders-startdistance)/2)
                        length = bodylmlist[24][2]-15
                        neck_point_xaxis = int(abs(bodylmlist[12][1]+bodylmlist[11][1])/2)
                        neck_point_yaxis = int(abs(bodylmlist[12][2]+length)/2)
                    else:
                        startdistance = None
            except:
                pass

                #-------------------------overlaying/Augmenting--------------------------------  
            try:
                #Scaling image
                heightImg, widthImg, _ = cloth.shape 
                newHeight, newWidth = ((heightImg + scaling) // 2)*2, ((widthImg + scaling) // 2)*2
                cloth = cv2.resize(cloth, (newWidth, newHeight))
                #Extracting alpha mask and coverting rgba into rgb
                b,g,r,a = cv2.split(cloth)
                rgb_img = cv2.merge((b,g,r))
                # Applying some simple filtering to remove edge noise
                mask = cv2.medianBlur(a,5)
                #ROI of background image
                regionOfInterest = img[neck_point_yaxis-newHeight//2:neck_point_yaxis+newHeight//2, neck_point_xaxis-newWidth//2:neck_point_xaxis+newWidth//2]
                #Black out the area behind region of interest in backgroud video
                bg_img = cv2.bitwise_and(regionOfInterest.copy(),regionOfInterest.copy(),mask = cv2.bitwise_not(mask))
                #masking out the forground image
                fg_img = cv2.bitwise_and(rgb_img,rgb_img,mask = mask)
                #updating the background video
                
                img[neck_point_yaxis-newHeight//2:neck_point_yaxis+newHeight//2, neck_point_xaxis-newWidth//2:neck_point_xaxis+newWidth//2] = cv2.add(bg_img, fg_img)
            except:
                pass
        cv2.imshow('AR TryOn',img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
arTryOn()
