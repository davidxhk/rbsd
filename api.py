import os
import shutil
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from lib import process_file, PARAMS

app = FastAPI(title="Snore Detection API")


@app.post("/detect")
async def detect_snore(
    file: UploadFile = File(...),
    fs_hz: int = Form(PARAMS["fs_hz"]),
    frame_ms: int = Form(PARAMS["frame_ms"]),
    hop_ms: int = Form(PARAMS["hop_ms"]),
    band_min_hz: int = Form(PARAMS["band_min_hz"]),
    band_max_hz: int = Form(PARAMS["band_max_hz"]),
    quiet_pct: int = Form(PARAMS["quiet_pct"]),
    quiet_delta: float = Form(PARAMS["quiet_delta"]),
    quiet_win_s: float = Form(PARAMS["quiet_win_s"]),
    dur_min_s: float = Form(PARAMS["dur_min_s"]),
    dur_max_s: float = Form(PARAMS["dur_max_s"]),
):
    # Validate file extension (scipy.io.wavfile only supports WAV)
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400, detail="Only .wav files are supported currently."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = process_file(
            tmp_path,
            fs_hz=fs_hz,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
            quiet_pct=quiet_pct,
            quiet_delta=quiet_delta,
            quiet_win_s=quiet_win_s,
            dur_min_s=dur_min_s,
            dur_max_s=dur_max_s,
        )
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error processing file {file.filename}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Check for dev flag
    dev_mode = "--dev" in sys.argv

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = "debug" if dev_mode else "info"

    print(f"Starting API server at {host}:{port} (Dev mode: {dev_mode})...")

    uvicorn.run(app, host=host, port=port, reload=dev_mode, log_level=log_level)
