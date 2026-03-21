from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
from fastapi.responses import PlainTextResponse
import os
import tempfile

app = FastAPI()

# Load model lazily on first request to keep container lightweight until used
_model = None
_model_name = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def models():
    return [{"id": "faster-whisper", "description": "Faster Whisper transcription via faster-whisper"}]

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: Optional[str] = Form("faster-whisper")):
    global _model, _model_name
    # Only faster-whisper supported in this container
    if model != "faster-whisper":
        return JSONResponse(status_code=400, content={"error": "unsupported model"})

    # Lazy load model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"failed to import faster-whisper: {e}"})
        # Use medium by default; set device to cpu
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
        _model_name = "faster-whisper:medium"

    # Save uploaded file to a temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        segments, info = _model.transcribe(tmp_path, beam_size=5)
        text = "".join([seg.text for seg in segments])
        return PlainTextResponse(text)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
