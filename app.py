import database
import json
from flask_cors import CORS
from flask import Flask,jsonify,request,abort
import bodymeasurements

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
    responseObj = bodymeasurements.measurements("6258b68a5992c70023c5724b")
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


    
if __name__ == "__main__":
    app.run(host='192.168.100.2',port=5000,debug=True)