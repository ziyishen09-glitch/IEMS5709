from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from typing import Any, Dict, List, Optional
import os
import tempfile

WORKING_MODEL = os.environ.get("WHISPER_MODEL_NAME", "medium")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "fp16")
DEFAULT_TASK = os.environ.get("WHISPER_TASK", "transcribe")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

app = FastAPI(title="Faster Whisper GPU ASR")

_model: Optional[WhisperModel] = None
_model_name = f"faster-whisper:{WORKING_MODEL}" if WORKING_MODEL else "faster-whisper"


def _ensure_model_loaded() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WORKING_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def _segment_to_dict(segment: Any) -> Dict[str, Any]:
    return {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": getattr(segment, "text", ""),
        "tokens": getattr(segment, "tokens", []),
    }


@app.on_event("startup")
async def startup_event():
    _ensure_model_loaded()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> List[Dict[str, str]]:
    return [
        {
            "id": "faster-whisper",
            "description": "Faster Whisper transcription via faster-whisper",
            "name": _model_name,
        }
    ]


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("faster-whisper"),
    task: str = Form(DEFAULT_TASK),
    language: Optional[str] = Form(None),
):
    if model != "faster-whisper":
        raise HTTPException(status_code=400, detail="unsupported model")

    whisper = _ensure_model_loaded()

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        tmp.write(await file.read())

    try:
        segments, info = whisper.transcribe(
            temp_path,
            beam_size=BEAM_SIZE,
            task=task,
            language=language,
        )
        text = "".join(segment.text for segment in segments).strip()
        return PlainTextResponse(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
