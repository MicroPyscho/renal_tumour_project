from flask import Flask, request, jsonify, render_template
import os
from flask import send_from_directory
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# ✅ Module level

@app.route('/')
def home():
    return send_from_directory('templates', 'index.html')
    

@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"


# And update predictRoute:
@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    image = request.json['image']
    clApp = ClientApp()
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
