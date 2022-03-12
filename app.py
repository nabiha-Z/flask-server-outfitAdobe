import database
import json
from flask_cors import CORS
from flask import Flask,jsonify,request,abort
import bodymeasurements

app = Flask(__name__)
cors=CORS()
cors.init_app(app)

@app.route("/", methods=['POST'])
def measurements():
    if not request.json :
        abort(400)
    print("Dksjdk")
    print(request.json['user'])
    responseObj = bodymeasurements.measurements(request.json['user'])
    print(responseObj)
    # print(responseObj.data)
    print(type(responseObj))
    if(responseObj['msg'] == "true"):

        print(type(responseObj['data']))
        val = json.dumps(responseObj['data'])
        print(val)
        print("type:", type(val))
        database.db.measurements.insert_one(responseObj['data'])
        print("Data added to database")
    else:
        print("not detected")
    return "Done"
    
if __name__ == "__main__":
    app.run(debug=True)