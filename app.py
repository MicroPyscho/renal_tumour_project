from flask import Flask, request, jsonify, render_template, abort
import os
from flask import send_from_directory
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import json
import yaml

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_from_directory("templates", "index.html")
 
 
@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"
 
 
@app.route("/scores")
@cross_origin()
def get_scores():
    """
    Serves scores.json (written by the evaluation stage) plus the
    current MODEL_NAME from params.yaml.  The frontend polls this
    every 8 s so metrics update automatically after every training run.
    """
    try:
        with open("scores.json") as f:
            s = json.load(f)
    except FileNotFoundError:
        return jsonify({
            "error": "scores.json not found — run the evaluation stage first"
        }), 404
 
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        s["model_name"] = params.get("MODEL_NAME", "unknown")
    except Exception:
        pass  # non-fatal
 
    return jsonify(s)
 
 
@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    """
    Accepts JSON body: {"image": "<base64-encoded JPEG>"}
 
    Returns:
      200  classification result  →  CT scan identified and classified
      400  gate error             →  image rejected (not a CT scan)
      500  unexpected error
    """
    try:
        image_b64 = request.json["image"]
 
        # Write to disk so PredictionPipeline can open it with PIL / Keras
        filename = "inputImage.jpg"
        decodeImage(image_b64, filename)
 
        pipeline = PredictionPipeline(filename)
        result   = pipeline.predict()
 
        # Gate failure → 400 so the frontend can show the specific reason
        if result and result[0].get("gate_failed"):
            return jsonify(result), 400
 
        return jsonify(result), 200
 
    except KeyError:
        return jsonify({"error": "Request body must contain an 'image' key"}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed — see server logs"}), 500
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)