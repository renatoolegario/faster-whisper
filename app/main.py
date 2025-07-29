from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile
import mimetypes

app = Flask(__name__)

# Configura CORS apenas para as origens permitidas
CORS(app, resources={
    r"/transcribe": {
        "origins": [
            "https://web.whatsapp.com",
            "https://faster-whisper.uaistack.com.br"
        ],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Limite de tamanho de arquivo: 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Carrega o modelo Whisper
print("🔊 Carregando modelo Whisper...")
try:
    model = whisper.load_model("base")
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo Whisper: {str(e)}")
    raise

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        print("⚠️ Nenhum arquivo enviado no campo 'file'")
        return jsonify({"error": "Nenhum arquivo enviado no campo 'file'"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        print("⚠️ Nome de arquivo inválido ou vazio")
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    # Verifica o tipo de arquivo (aceita apenas áudio)
    mime_type, _ = mimetypes.guess_type(file.filename)
    if not mime_type or not mime_type.startswith('audio/'):
        print(f"⚠️ Tipo de arquivo inválido: {file.filename} (mime: {mime_type})")
        return jsonify({"error": "Apenas arquivos de áudio são permitidos"}), 400

    print(f"📥 Arquivo recebido: {file.filename} (tipo: {mime_type})")
    audio_path = None

    try:
        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            file.save(tmp.name)
            audio_path = tmp.name

        print(f"🎧 Transcrevendo: {audio_path}")
        result = model.transcribe(audio_path)
        transcription = result["text"].strip()
        print(f"📝 Transcrição concluída: {transcription}")
        return jsonify({"text": transcription})

    except Exception as e:
        print(f"❌ Erro durante transcrição: {str(e)}")
        return jsonify({"error": f"Erro ao processar o áudio: {str(e)}"}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"🧹 Arquivo temporário removido: {audio_path}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)