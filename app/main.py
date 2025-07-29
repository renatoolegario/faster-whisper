from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)
model = whisper.load_model("base")  # Pode trocar para "small", "medium", etc.

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file.save(tmp.name)
        try:
            result = model.transcribe(tmp.name)
            return jsonify({"text": result["text"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp.name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
