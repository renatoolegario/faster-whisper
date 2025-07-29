from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()
model = WhisperModel("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp.write(await file.read())
        segments, _ = model.transcribe(tmp.name)
        result = " ".join(segment.text for segment in segments)
    return {"transcription": result}