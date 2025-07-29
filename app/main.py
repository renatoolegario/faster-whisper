import base64
from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)
model = whisper.load_model("large-v3")  # ou "medium", "base"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" in request.files:
        # multipart/form-data (continua funcionando)
        file = request.files["file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            file.save(tmp.name)
            audio_path = tmp.name

    elif request.is_json and "base64" in request.json:
        # JSON com Base64
        try:
            audio_data = base64.b64decode(request.json["base64"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(audio_data)
                audio_path = tmp.name
        except Exception as e:
            return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400
    else:
        return jsonify({"error": "No file or base64 provided"}), 400

    try:
        result = model.transcribe(audio_path)
        return jsonify({"text": result["text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(audio_path)
