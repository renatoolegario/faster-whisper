from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Importa CORS
import whisper
import os
import tempfile

app = Flask(__name__)
CORS(app)  # ✅ Habilita CORS para todas as rotas

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Limite: 100MB

print("🔊 Carregando modelo Whisper...")
model = whisper.load_model("base")
print("✅ Modelo carregado com sucesso!")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado no campo 'file'"}), 400

    file = request.files["file"]
    print(f"📥 Arquivo recebido: {file.filename}")
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            file.save(tmp.name)
            audio_path = tmp.name

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
