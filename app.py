import database
import json
from flask_cors import CORS
from flask import Flask,jsonify,request,abort
import bodymeasurements
import mobileMeasurement
import arTryon
import dressTryOn
import mobileARTryOn
import mobileBottomTryOn
import mobileDressTryOn
import bottomTryOn
import os

# from pathlib import Path
# import tempfile
import cv2
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'E:/FYP/flask-server-outfitAdobe/dresses'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'trial_dresses')
    if not os.path.isdir(target):
        os.mkdir(target)
    print("welcome to upload`")
    filename= request.args['filename']
    filename = filename + '.png'
    file = request.files['file']

    # filename = request.files['filename']
   
    destination="/".join([target, filename])
    # print(destination)
    file.save(destination)
    # session['uploadFilePath']=destination
    response="Whatever you wish too return"
    return "true"

@app.route("/home", methods=['GET'])
def home():
    
    print("Dksjdk")
    return "true"

@app.route("/measurements", methods=['POST'])
def measurements():
    if not request.json :
        abort(400)
    print("Dksjdk")
    print(request.json['user'])
    responseObj = bodymeasurements.measurements(request.json['user'])
    print(responseObj)
    # print(responseObj.data)
    print(type(responseObj))
    check = "false"
    if(responseObj['msg'] == "true"):

        print(type(responseObj['data']))
        val = json.dumps(responseObj['data'])
        print(val)
        check="true"
        print("type:", type(val))
        database.db.measurements.insert_one(responseObj['data'])
        print("Data added to database")
    else:
        print("not detected")
    return check

@app.route("/mobilemeasurements", methods=['POST'])
def mobilemeasurements():
    print("Dksjdk")
    user = request.args['user']
    print("user: ", user)
    # print(request.files['mobile-video-upload'])
    print("file: ",request.files['video'])
    uploaded_file=request.files['video']
    responseObj = mobileMeasurement.mobilemeasurements(uploaded_file,user)
    print("response: ",responseObj)
    
    # print(uploaded_file.read())
    # with tempfile.TemporaryDirectory() as td:
    #     temp_filename = Path(td) / 'uploaded_video'
    #     uploaded_file.save(temp_filename)
    # print("uploaded file name: ",uploaded_file.name)
    # uploaded_file.save('E:/FYP/flask-server-outfitAdobe/user_videos/video.mp4')
    # print("hello")
    # vidcap = cv2.VideoCapture('E:/FYP/flask-server-outfitAdobe/user_videos/video.mp4')
    # while (vidcap.isOpened()):
    #     success, image= vidcap.read()
        
        
    #     # Recolor back to BGR     
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('Meausrement',image)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    # vidcap.release()
    # cv2.destroyAllWindows()
    # responseObj = mobilemeasurements.mobilemeasurements(request.json['videoUri'])
    # print(responseObj)
    # # print(responseObj.data)
    # print(type(responseObj))
    # bodymeasurement = {
    #         'shoulders' :12,
    #         'fullLength':21,
    #         'arms':22,
    #         "knee": 14,
    #         "tshirt":41,
    #         'bottom':44,
    #         "waist":44,
    #         "user":user.replace('"','')
    #     }
    # check="true"
    # database.db.measurements.insert_one(bodymeasurement)
    # print("Data added to database")


    check = "false"
    if(responseObj['msg'] == "true"):

        print(type(responseObj['data']))
        val = json.dumps(responseObj['data'])
        print(val)
        check="true"
        database.db.measurements.insert_one(responseObj['data'])
        print("Data added to database")
    else:
        print("not detected")
    return check

# user.replace('"','')
@app.route("/arTryOn", methods=['POST'])
def artryon():
    
    print(request.json['dress'])
    dress = request.json['dress']
    flag = request.json['flag']
    print(flag)
    dressPath ="dresses/trial_dresses/"+dress+".png"
    print("path: ", dressPath)
    if(flag == 0):
        response = arTryon.arTryOn(dressPath)
    elif(flag == 1):
            response = bottomTryOn.bottomTryOn(dressPath)
    elif(flag == 2):
            response = dressTryOn.dressTryOn(dressPath)
    return "true"

@app.route("/mobileArTryOn", methods=['POST'])
def mobileartryon():
    

    dress = request.args['dress']
    flag = request.args['flag']
    print("flag: ",flag)
    dressPath ="dresses/trial_dresses/"+dress+".png"
    print("path: ", dressPath)
    uploaded_file=request.files['video']
    # print("file: ", uploaded_file)
    if(flag == 0):
         response = mobileARTryOn.mobileTryOn(uploaded_file,dressPath)
    elif(flag == 1):
         response = mobileBottomTryOn.mobileBottomTryOn(uploaded_file,dressPath)
    else:
        response = mobileDressTryOn.mobileDressTryOn(uploaded_file,dressPath)
    return "true"

if __name__ == "__main__":
    app.run(host='192.168.100.8',port=5000,debug=True)