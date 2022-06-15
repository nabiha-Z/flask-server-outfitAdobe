import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone
import math
import time
import imutils
import mediapipe as mp

# stopAR = False
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
# startdistance = None
# neck_point_xaxis = 500
# neck_point_yaxis = 500
# scaling = 0
# start_time = time.time()
# capture_duration = 30
def calculate_angle(L,R):
   
    s2=L[1]
    t2=L[2]
    s1=R[1]
    t1=R[2]
    shoulderslength = ((math.hypot(s2 - s1, t2 - t1)) )
    shoulderslength = round(shoulderslength)
    print(shoulderslength)
    return shoulderslength

def mobileTryOn(uploaded_file,dress):
    stopAR = False
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    startdistance = None
    neck_point_xaxis = 500
    neck_point_yaxis = 500
    scaling = 0
    start_time = time.time()
    capture_duration = 40
    # print("uploaded file name: ",uploaded_file.name)
    # uploaded_file.save('user_videos/video.mp4')
    print("capture")
    cap = cv2.VideoCapture(0)
    cap.set(3,1300)   #width
    cap.set(4,950)    #height
    cap.set(10,200)
    takeSS = False
    detector = HandDetector(detectionCon=0.8)
    length = 1000

    while (int(time.time() - start_time) < capture_duration):
        success, img = cap.read()
        # img = imutils.resize(img, width=350)
        #1 means horizontal 
        
        img = cv2.flip(img,1)
        h, w, c = img.shape
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img = cv2.putText(img, "TryOn Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(252, 102, 75), 2)
        if stopAR == False:
            
            cloth = cv2.imread(dress, cv2.IMREAD_UNCHANGED)
            cloth = cv2.resize(cloth, (430,540))
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
                    length = calculate_angle(bodylmlist[11],bodylmlist[12]) 
                    print("length: ", length)
                    if(length<140):
                        
                        cloth = cv2.imread('dresses/sleeve1.png', cv2.IMREAD_UNCHANGED)
                        cloth = cv2.resize(cloth, (450,460))
                        print("rotateeeee")
              
                        LXP1=(bodylmlist[11][1])-0.09
                        LYP2=(bodylmlist[11][2])
                        LS1=LXP1,LYP2
                        LXP3=(bodylmlist[11][1])+0.09
                        LYP4=(bodylmlist[11][2])
                        LS2=LXP3,LYP4
                       
                        if startdistance is None:
                            print("sasadsadad")
                            length_of_shoulders = abs(LXP3-LXP1)
                            startdistance = length_of_shoulders
                            length_of_shoulders = abs(LXP3-LXP1)
                            scaling = int((length_of_shoulders-startdistance)/2)
                            length = bodylmlist[25][2]-170
                            neck_point_xaxis = int(abs(LXP3+LXP1)/2)
                            neck_point_yaxis = int(abs(LYP4+length)/2)
                        else:
                            print("None")
                            startdistance = None
                        

                    else:    
                        if startdistance is None:
                            length_of_shoulders = abs(bodylmlist[12][1]-bodylmlist[11][1])
                            startdistance = length_of_shoulders
                            length_of_shoulders = abs(bodylmlist[12][1]-bodylmlist[11][1])
                            scaling = int((length_of_shoulders-startdistance)/2)
                            length = bodylmlist[26][2]-255
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
mobileTryOn('TestingVideo7.mp4','dresses/shirtt.png')