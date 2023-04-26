from flask import Flask,jsonify, request, url_for, abort
from predict_image import Predictor
app = Flask(__name__)

predictor = Predictor()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_path = data["path"]
    response = predictor.predict(img_path)
    response["status"] = "Success"
    return response