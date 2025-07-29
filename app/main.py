from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
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

# ✅ Configuração otimizada
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Dispositivo: {device}")

# Carrega o modelo Whisper Medium
print("🔊 Carregando modelo Whisper MEDIUM...")
try:
    model = whisper.load_model("small", device=device)
    print("✅ Modelo MEDIUM carregado com sucesso!")
    print(f"📊 Parâmetros: 769M | Dispositivo: {device}")
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

    # ✅ Verificar tamanho do arquivo
    file.seek(0, 2)  # Vai para o final
    file_size = file.tell()
    file.seek(0)  # Volta para o início
    
    if file_size == 0:
        print("⚠️ Arquivo está vazio")
        return jsonify({"error": "Arquivo de áudio está vazio"}), 400

    # Verifica o tipo de arquivo (aceita apenas áudio)
    mime_type, _ = mimetypes.guess_type(file.filename)
    if not mime_type or not mime_type.startswith('audio/'):
        print(f"⚠️ Tipo de arquivo inválido: {file.filename} (mime: {mime_type})")
        return jsonify({"error": "Apenas arquivos de áudio são permitidos"}), 400

    print(f"📥 Arquivo recebido: {file.filename} | Tamanho: {file_size/1024:.1f} KB | Tipo: {mime_type}")
    audio_path = None

    try:
        # ✅ Detectar formato pelos primeiros bytes
        header = file.read(12)
        file.seek(0)
        
        file_extension = ".ogg"  # padrão
        if header.startswith(b'OggS'):
            file_extension = ".ogg"
        elif header.startswith(b'RIFF'):
            file_extension = ".wav"
        elif header[4:8] == b'ftyp':
            file_extension = ".m4a"
        elif header.startswith(b'\x1a\x45\xdf\xa3'):
            file_extension = ".webm"
        
        print(f"🔍 Formato detectado: {file_extension}")

        # Salva o arquivo temporariamente com extensão correta
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            file.save(tmp.name)
            audio_path = tmp.name

        # Verificar se arquivo foi salvo corretamente
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Arquivo temporário não foi salvo corretamente")

        print(f"🎧 Transcrevendo com MEDIUM: {audio_path} ({os.path.getsize(audio_path)} bytes)")
        
        # ✅ Transcrição otimizada para português
        result = model.transcribe(
            audio_path,
            language="pt",           # Forçar português brasileiro
            task="transcribe",       # Transcrição (não tradução)
            fp16=(device == "cuda"), # fp16 para GPU, float32 para CPU
            temperature=0.0,         # Determinístico
            beam_size=5,            # Melhor qualidade
            best_of=5,              # Múltiplas tentativas
            patience=1.0,           # Paciência na busca
            suppress_tokens=[-1],   # Suprimir tokens especiais
            initial_prompt="Transcrição em português brasileiro:",
            word_timestamps=False   # Mais rápido sem timestamps
        )
        
        transcription = result["text"].strip()
        
        # ✅ Limpeza básica do texto
        transcription = transcription.replace("  ", " ")  # Remover espaços duplos
        if transcription.startswith("Transcrição em português brasileiro:"):
            transcription = transcription.replace("Transcrição em português brasileiro:", "").strip()
        
        print(f"📝 Transcrição concluída ({len(transcription)} chars): {transcription[:100]}...")
        
        return jsonify({
            "text": transcription,
            "language": result.get("language", "pt"),
            "duration": round(result.get("duration", 0), 2)
        })

    except Exception as e:
        print(f"❌ Erro durante transcrição: {str(e)}")
        return jsonify({"error": f"Erro ao processar o áudio: {str(e)}"}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"🧹 Arquivo temporário removido: {audio_path}")

@app.route("/health", methods=["GET"])
def health():
    """Endpoint para verificar status da API"""
    return jsonify({
        "status": "ok",
        "model": "small",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0"
    })

if __name__ == "__main__":
    print("🌐 Servidor Whisper Medium iniciado!")
    print("🎯 Otimizado para português brasileiro")
    print("🔗 Acesse: http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=False)