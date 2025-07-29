import base64
from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Limite: 100MB

print("🔊 Carregando modelo Whisper...")
model = whisper.load_model("base")
print("✅ Modelo carregado com sucesso!")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_path = None

    try:
        if "file" in request.files:
            file = request.files["file"]
            print(f"📥 Arquivo recebido: {file.filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                file.save(tmp.name)
                audio_path = tmp.name

        elif request.is_json and "base64" in request.json:
            try:
                print("📥 Recebido base64 via JSON")
                audio_data = base64.b64decode(request.json["base64"])
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                    tmp.write(audio_data)
                    audio_path = tmp.name
            except Exception as e:
                return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400

        else:
            return jsonify({"error": "No file or base64 provided"}), 400

        print(f"🎧 Transcrevendo: {audio_path}")
        result = model.transcribe(audio_path)
        print(f"📝 Transcrição concluída: {result['text']}")
        return jsonify({"text": result["text"]})

    except Exception as e:
        print(f"❌ Erro durante transcrição: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"🧹 Arquivo temporário removido: {audio_path}")


# 🔥 IMPORTANTE: isso garante que o servidor Flask rode no Docker corretamente
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
