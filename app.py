import database
import json
from flask_cors import CORS
from flask import Flask,jsonify,request,abort
import bodymeasurements
import mobileMeasurement
import arTryon
import bottomTryOn
# from pathlib import Path
# import tempfile
import cv2
app = Flask(__name__)
cors=CORS()
cors.init_app(app)

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

@app.route("/arTryOn", methods=['POST'])
def artryon():
    print("reques: ", request)
    print("Try on", request.files)
    dress = request.files['dress']
    print("dress: ", dress)
    # print(request.files['mobile-video-upload'])

    # responseObj = arTryOn.tryOn(dress)  


    
if __name__ == "__main__":
    app.run(host='192.168.100.8',port=5000,debug=True)