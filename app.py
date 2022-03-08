from flask_cors import CORS
from flask import Flask,jsonify,request
import bodymeasurements

app = Flask(__name__)
cors=CORS()
cors.init_app(app)

@app.route("/")
def measurements():
    shoulders = bodymeasurements.measurements()
    msg = "Shoulders Length: "+ str(shoulders) + ""
    return msg
    
if __name__ == "__main__":
    app.run(debug=True)