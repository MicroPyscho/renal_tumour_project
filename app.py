import os
import json
import uuid
import yaml
import shutil
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.pipeline.viz_prediction import VisPredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# ── Session store: maps session_id → {model_path, class_names,
#                                      preprocess, img_size, tmp_dir}
SESSION_STORE: dict = {}


# ─────────────────────────────────────────────────────────────────
# CORE AIDRENAL ROUTES
# ─────────────────────────────────────────────────────────────────

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
        with open(os.path.join(os.path.dirname(__file__), "scores.json")) as f:
            s = json.load(f)
    except FileNotFoundError:
        return jsonify({
            "error": "scores.json not found — run the evaluation stage first"
        }), 404

    try:
        with open(os.path.join(os.path.dirname(__file__), "params.yaml")) as f:
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
        filename = os.path.join("/tmp", "inputImage.jpg")
        decodeImage(image_b64, filename)

        # Read the model selector choice from the frontend (optional)
        requested_model = request.json.get("model", None)

        pipeline = PredictionPipeline(filename)
        result   = pipeline.predict(requested_model=requested_model)

        # Gate failure → 400 so the frontend can show the specific reason
        if result and result[0].get("gate_failed"):
            return jsonify(result), 400

        return jsonify(result), 200

    except KeyError:
        return jsonify({"error": "Request body must contain an 'image' key"}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed — see server logs"}), 500

# VISUALISE YOUR MODEL ROUTES
@app.route("/vis-upload", methods=["POST"])
@cross_origin()
def vis_upload():
    """
    Accepts a visitor's model file and config.
    Stores it in a per-session temp directory.
    Returns a session_id the browser sends with every /vis-predict call.
    """
    try:
        if "model" not in request.files:
            return jsonify({"error": "No model file provided"}), 400

        model_file  = request.files["model"]
        class_names = json.loads(request.form.get("class_names", "[]"))
        preprocess  = request.form.get("preprocess",  "divide255")
        img_size    = int(request.form.get("img_size", 224))

        if not class_names:
            return jsonify({"error": "class_names must be a non-empty list"}), 400

        ext = os.path.splitext(model_file.filename)[1].lower()
        if ext not in (".keras", ".h5"):
            return jsonify({"error": "Model must be .keras or .h5"}), 400

        # Create isolated temp directory for this session
        session_id = str(uuid.uuid4())
        tmp_dir    = tempfile.mkdtemp(prefix=f"vis_{session_id[:8]}_")
        model_path = os.path.join(tmp_dir, f"model{ext}")
        model_file.save(model_path)

        SESSION_STORE[session_id] = {
            "model_path":  model_path,
            "class_names": class_names,
            "preprocess":  preprocess,
            "img_size":    img_size,
            "tmp_dir":     tmp_dir,
        }

        return jsonify({
            "session_id":  session_id,
            "class_names": class_names,
            "preprocess":  preprocess,
            "img_size":    img_size,
            "model_size_mb": round(os.path.getsize(model_path) / 1024 / 1024, 1),
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Upload failed — see server logs"}), 500


@app.route("/vis-predict", methods=["POST"])
@cross_origin()
def vis_predict():
    """
    Runs a single forward pass using the visitor's session model.
    Body: {"image": "<base64 JPEG>", "session_id": "<uuid>"}
    """
    try:
        data       = request.get_json(force=True)
        session_id = data.get("session_id", "")
        image_b64  = data.get("image", "")

        if session_id not in SESSION_STORE:
            return jsonify({"error": "Session not found or expired — re-upload your model"}), 404

        session = SESSION_STORE[session_id]

        # Save image to the session's temp dir
        img_path = os.path.join(session["tmp_dir"], "input.jpg")
        decodeImage(image_b64, img_path)

        pipeline = VisPredictionPipeline(
            filename    = img_path,
            model_path  = session["model_path"],
            class_names = session["class_names"],
            preprocess  = session["preprocess"],
            img_size    = session["img_size"],
        )
        result = pipeline.predict()

        if result and result[0].get("gate_failed"):
            return jsonify(result), 400
        return jsonify(result), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed — see server logs"}), 500


@app.route("/vis-clear", methods=["POST"])
@cross_origin()
def vis_clear():
    """Deletes the session model file and clears state."""
    try:
        session_id = request.get_json(force=True).get("session_id", "")
        if session_id in SESSION_STORE:
            tmp_dir = SESSION_STORE.pop(session_id).get("tmp_dir", "")
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"cleared": True}), 200
    except Exception:
        return jsonify({"error": "Clear failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)