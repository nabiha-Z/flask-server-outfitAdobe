import database
from flask_cors import CORS
from flask import Flask,jsonify,request
import bodymeasurements

app = Flask(__name__)
cors=CORS()
cors.init_app(app)

@app.route("/")
def measurements():
    print("Dksjdk")
    responseObj = bodymeasurements.measurements()
    print(responseObj)
    # database.db.user_collection.insert_one(responseObj.data);
    # print("Data added to database")
    return "Done"
    
if __name__ == "__main__":
    app.run(debug=True)