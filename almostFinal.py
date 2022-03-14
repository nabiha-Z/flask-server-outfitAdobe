# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:29:12 2021

@author: noorf
"""

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time
import qrcode
import pyautogui
import requests
import os
import face_recognition
import mediapipe as mp

CustomerShopping = None
Products = []
productImagesFound = []
productToAugment = None
productIterator = 0
# print("△"=="▽")
# body = {'customer' : CustomerShopping , 'productId' : productToAugment}
# requests.post('http://localhost:5000/carts/add', data = body)

#-----------------------------------------------------------------------------------------------------
#get face image names of all previous customers
# path = 'F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\customers'
# listOfImages = os.listdir(path)  #has images name in the list
# # print(listOfImages)
cusImagesFound = []

#iterate over all images found, i is image name and load actual images
# for i in range(len(listOfImages)):
#     # print(listOfImages[i])
#     image = cv2.imread(path+'\\'+listOfImages[i])
#     cusImagesFound.append(image) 
    
#-----------------------------------------------------------------------------------------------------
#after loading imgaes, detect locate faces in those images and store their encodings (features and points)
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

#-------------------------------------------------------------------------------------------------
# def matchCustomerImage():
#     encodeListKnown = findEncodings(cusImagesFound)
#     print("Encoding Complete")
#     imgTest = cv2.imread("known1.png")
#     imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#     faceLocTest = face_recognition.face_locations(imgTest)
#     encodeTest = face_recognition.face_encodings(imgTest, faceLocTest)
#     for encodeFace,faceLoc in zip(encodeTest,faceLocTest ):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print("faceDis ",faceDis)
#         matchIndex = np.argmin(faceDis)
        
#         if faceDis[matchIndex] < 0.55:
#             CustomerShopping = os.path.splitext(listOfImages[matchIndex])[0]
#             print("inside if: ", CustomerShopping)
#             break
#         else:
#             postFacePicture()
#             break
        
#-----------------------------------------------------------------------------------------------------
def postFacePicture():
    picturePosted = False
    payload = {'picture': ('new.png', open('customer.png','rb'))}  #first arg is filename sent to server, second is actual file/ picture
    res = requests.post('http://localhost:5000/customers/add', files = payload) #donot change files word
    CustomerShopping = (res.text).replace('"',"") #replace "" with empty
    print("customer posted ",CustomerShopping)
    
    picturePosted = True
    if picturePosted:
        os.rename('F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\customers\\new.png','F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\customers\\'+CustomerShopping+'.png')
    return 1

#-----------------------------------------------------------------------------------------------------
path = 'F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\categories'
categoryImages = os.listdir(path)  #has images name in the list
# print("categories",categoryImages)
catImagesFound = []
for i in range(len(categoryImages)):
    image = cv2.imread(path+'\\'+categoryImages[i],cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (350,350))
    catImagesFound.append(image) 
# cv2.resize(image,(0,0),None,0.25,0.25)  #resize image by scaler

#-----------------------------------------------------------------------------------------------------
def getProducts(Products):
    for i in range(len(Products)):
        image = cv2.imread('F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\products\\'+Products[i]+'.png',cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (350,350))
        productImagesFound.append(image) 
# cv2.resize(image,(0,0),None,0.25,0.25)  #resize image by scaler

#-----------------------------------------------------------------------------------------------------
def takeTrialScreenshot():
      # image = pyautogui.screenshot(region=(0,0, 300, 400))
      pyautogui.screenshot("SmaartShot.png")
      files = {'picture': ('SmaartShot.png', open('SmaartShot.png','rb'))}  #first arg is filename sent to server, second is actual file/ picture
      res = requests.post('http://localhost:5000/qrcode/uploadsingle', files = files)
      _id = (res.text).replace('"',"") #replace "" with empty
      qr = qrcode.QRCode(box_size=10)
      qr.add_data(_id)
      qr.make()
      img_qr = qr.make_image()
      img_qr.save('SmaartQr.png')
      return 1
  
#-----------------------------------------------------------------------------------------------------
logo = cv2.imread("SMAART logo.png", cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (180,100))
code = cv2.imread("qr code.png", cv2.IMREAD_UNCHANGED)
screen = "start"
actions = ["Start Shopping","Back","^","Try Item", "Add to Cart", "Remove","Click Picture", "Delete Cart", "Checkout", "Men", "Women", "Kids", "Unisex","^","V"]
showQR = 0
# print(actions[13])

def reset():
    global boxes
    boxes = [[0,0,0,0]] * 15
    global action
    action = None


def clicked(cursor, boxes):
    for i, bbox in enumerate(boxes, start=0):
        x1, y1, x2, y2 = bbox
        # [0] is x cord [1] is y
        if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:   
            # print(i)
            global action
            action = actions[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), (77,0,77), cv2.FILLED)
            
#-----------------------------------------------------------------------------------------------------
stopAR = False
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
startdistance = None
neck_point_xaxis = 500
neck_point_yaxis = 500
scaling = 0


#---------------------------------------------------------------MAIN VIDEO STREAM LOOP----------------------------------------------------------------------------

vid = cv2.VideoCapture(0) #0 for irium 1 for webcam
vid.set(3,1300)   #width
vid.set(4,950)    #height
vid.set(10,200)
takeSS = False
detector = HandDetector(detectionCon=0.8)

while (vid.isOpened()):
    success, img = vid.read()
    #1 means horizontal 
    img = cv2.flip(img,1)
    
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    img = cvzone.overlayPNG(img, logo, [1100,850])
    
    imgS = cv2.resize(img,(0,0),None,0.5,0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    if (facesCurFrame and takeSS):
        pyautogui.screenshot("customer.png")
        # matchCustomerImage()
        print("done")
        takeSS = False
  
    hands, img = detector.findHands(img,flipType=False)
    
    if screen == "start":
        reset()
        if action == "checkout":
            takeSS = True
        img, bbox0 = cvzone.putTextRect(img, actions[0] , [500, 450], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))  #selection option box
        boxes.insert(0,bbox0)
        
    if screen == "category":
        reset()
        img = cv2.putText(img, "Category Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img, bbox1 = cvzone.putTextRect(img, actions[1], [1100, 100], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77)) #change scale to 2 to increase size a lil 
        boxes.insert(1,bbox1)
        img = cvzone.overlayPNG(img, catImagesFound[0], [100,100])
        img, bbox9 = cvzone.putTextRect(img, actions[9], [240, 500], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(9,bbox9)
        img = cvzone.overlayPNG(img, catImagesFound[1], [700,100])
        img, bbox10 = cvzone.putTextRect(img, actions[10], [815, 500], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(10,bbox10)
        img = cvzone.overlayPNG(img, catImagesFound[2], [100,550])
        img, bbox11 = cvzone.putTextRect(img, actions[11], [240, 900], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(11,bbox11)
        img = cvzone.overlayPNG(img, catImagesFound[3], [700,550])
        img, bbox12 = cvzone.putTextRect(img, actions[12], [815, 900], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(12,bbox12)
    
    if screen == "product":
        reset()
        img = cv2.putText(img, "Product Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img, bbox1 = cvzone.putTextRect(img, actions[1], [1050, 150], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))  
        boxes.insert(1,bbox1)
        img, bbox13 = cvzone.putTextRect(img, actions[13], [250, 800], 2, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 30, 2,(77,0,77))  
        boxes.insert(13,bbox13)
        img, bbox2 = cvzone.putTextRect(img, actions[2], [250, 200], 2, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 30, 2,(77,0,77))  
        boxes.insert(2,bbox2)
        img = cvzone.overlayPNG(img, productImagesFound[productIterator], [100,300])
        img = cv2.putText(img, (Products[productIterator]['name']) , (470,450),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img = cv2.putText(img, "Rs. "+str(Products[productIterator]['price']) , (470,500),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img = cv2.putText(img, (Products[productIterator]['description']) , (470,550),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img, bbox3 = cvzone.putTextRect(img, actions[3], [650, 620], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(3,bbox3)
        # img = cvzone.overlayPNG(img, productImagesFound[1], [100,550])
        # img = cv2.putText(img, (Products[1]['name']), (470,700),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        # img = cv2.putText(img, "Rs. "+str(Products[1]['price']) , (470,750),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        # img = cv2.putText(img, (Products[1]['description']) , (470,800),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        # img, bbox3 = cvzone.putTextRect(img, actions[3], [650, 870], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        # # boxes.insert(3,bbox3) 
        productToAugment = Products[productIterator]['id']
        # print(productToAugment)
        
    if screen == "ar try":
        reset()
        img = cv2.putText(img, "Trial Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        if stopAR == False and productToAugment:
            print(productToAugment)
            cloth = cv2.imread('F:\\SMAART complete\\SMAART-final-project-POS\\back-end\\products\\'+productToAugment+'.png', cv2.IMREAD_UNCHANGED)
            cloth = cv2.resize(cloth, (400,400))
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
                    neck_point_xaxis = int(abs(bodylmlist[12][1]+bodylmlist[11][1])/2)
                    neck_point_yaxis = int(abs(bodylmlist[12][2]+bodylmlist[24][2])/2)
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
        img, bbox1 = cvzone.putTextRect(img, actions[1], [1050, 150], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))  
        boxes.insert(1,bbox1)
        img, bbox4 = cvzone.putTextRect(img, actions[4], [200, 400], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(4,bbox4)
        img, bbox6 = cvzone.putTextRect(img, actions[6], [200, 650], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(6,bbox6)
        img, bbox8 = cvzone.putTextRect(img, actions[8], [550, 800], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(8,bbox8)
        
    if screen == "cart":
        stopAR = True
        reset()
        img = cv2.putText(img, "Cart Screen", (500,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img, bbox1 = cvzone.putTextRect(img, actions[1], [1050, 150], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))  
        boxes.insert(1,bbox1)
        img, bbox5 = cvzone.putTextRect(img, actions[5], [200, 400], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(5,bbox5)
        img, bbox7 = cvzone.putTextRect(img, actions[7], [200, 500], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(7,bbox7)
        img, bbox8 = cvzone.putTextRect(img, actions[8], [550, 800], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))
        boxes.insert(8,bbox8)
        
    if screen == "scan code":
        stopAR = True
        reset()
        img = cv2.putText(img, "Scan the below QR code to get Picture", (300,80),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
        img, bbox1 = cvzone.putTextRect(img, actions[1], [1100, 100], 1, 2, (77,0,77),(255,255,255), cv2.FONT_HERSHEY_COMPLEX , 15, 2,(77,0,77))  
        boxes.insert(1,bbox1)
        if(showQR == 1):
            img[320:650,485:815] = cv2.imread("SmaartQr.png")
        else:
            img = cv2.putText(img, "Picture not found. Try again", (500,500),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
    
    if hands:
        handCoordinates = hands[0]['lmList']  #first hand detected
        cursor = handCoordinates[8]   #from mediapipe // index finger tip is 8, cursor 
        distance, details  = detector.findDistance(handCoordinates[8], handCoordinates[4])   #4 is tip of thumb
        
        fingers = detector.fingersUp(hands[0])
        fingerCount = sum(fingers)
        if boxes and fingerCount == 5:
            clicked(cursor,boxes)
            if action is not None:
                print(action)
                time.sleep(0.15)
                
                if action == "^":
                    print("up")
                    if productIterator == 0:
                        productIterator = (len(Products)-1)
                    else:
                        productIterator -= 1
                        
                if action == "V":
                    print("down")
                    if productIterator == (len(Products)-1):
                        productIterator = 0
                    else:
                        productIterator += 1
                print(productIterator)
    
        elif boxes and distance < 25:
            clicked(cursor, boxes)
            if action is not None:
                time.sleep(0.15)
                
                if action == "Start Shopping":
                    screen = "category"
                    
                if action == "Men":
                    Products = []
                    productImagesFound = []
                    data = requests.get('http://localhost:5000/products/cat/Men').json()
                    for i in data:
                        Products.append({"id": i['_id'],"name": i['name'],"price": i['price'],"description": i['description']})
                    getProducts([x['id'] for x in Products])
                    screen = "product"
                    
                if action == "Women":
                    Products = []
                    productImagesFound = []
                    data = requests.get('http://localhost:5000/products/cat/Women').json()
                    for i in data:
                        Products.append({"id": i['_id'],"name": i['name'],"price": i['price'],"description": i['description']})
                    getProducts([x['id'] for x in Products])
                    screen = "product"
                    
                if action == "Kids":
                    Products = []
                    productImagesFound = []
                    data = requests.get('http://localhost:5000/products/cat/Kids').json()
                    for i in data:
                        Products.append({"id": i['_id'],"name": i['name'],"price": i['price'],"description": i['description']})
                    getProducts([x['id'] for x in Products])
                    screen = "product"
                    
                if action == "Unisex":
                    Products = []
                    productImagesFound = []
                    data = requests.get('http://localhost:5000/products/cat/Unisex').json()
                    for i in data:
                        Products.append({"id": i['_id'],"name": i['name'],"price": i['price'],"description": i['description']})
                    getProducts([x['id'] for x in Products])
                    screen = "product"
                
                if action == "Select Item":
                    screen = "product"
                    
                if action == "Add to Cart":
                    stopAR = True
                    body = {'customer' : CustomerShopping , 'productId' : productToAugment}
                    requests.post('http://localhost:5000/carts/add', data = body)
                    screen = "cart"
                    
                if action == "Try Item":
                    productToAugment = Products[1]['id']
                    screen = "ar try"
                    stopAR = False
                    
                if action == "Click Picture":
                    stopAR = True
                    showQR = takeTrialScreenshot()
                    screen = "scan code"
                    
                if action == "Remove":
                    body = {'customer' : CustomerShopping , 'productId' : productToAugment}
                    requests.post('http://localhost:5000/carts/remove', data = body)
                    screen = "product"
                    
                if action == "Delete Cart":
                    body = {'customer' : CustomerShopping}
                    requests.post('http://localhost:5000/carts/empty', data = body)
                    screen = "product"
                    
                if action == "Back" and screen == "category":
                    screen = "start"
                    
                if action == "Back" and screen == "product":
                    screen = "category"
                    
                if action == "Back" and screen == "ar try":
                    stopAR = True
                    screen = "product"
                    
                if action == "Back" and screen == "cart":
                    screen = "product"  
                
                if action == "Back" and screen == "scan code":
                    stopAR = False
                    screen = "ar try"
                    
                if action == "Checkout":
                    stopAR = True
                    img = cv2.putText(img, "Checkout Successful", (500,500),cv2.FONT_HERSHEY_COMPLEX , 1,(77,0,77), 2)
                    time.sleep(0.2)
                    screen = "start"
    
    cv2.imshow('Smart Mirror',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()



