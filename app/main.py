from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import AudioError, Detector
from .sources import ResolvedAudio, SourceError, resolve_link

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}

app = FastAPI(title="AI Music Scanner")
detector = Detector()


def _analyze(resolved: ResolvedAudio) -> dict:
    result = detector.analyze(str(resolved.path)).to_dict()
    result.update(
        source=resolved.source,
        title=resolved.title,
        detail=resolved.detail,
        cover_url=resolved.cover_url,
        notes=resolved.notes,
    )
    return result


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Dateityp {ext or '?'} wird nicht unterstützt")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{ext}"
        size = 0
        with path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Datei ist zu groß (max. 200 MB)")
                f.write(chunk)
        try:
            return await run_in_threadpool(_analyze, ResolvedAudio(path, "upload", title=file.filename))
        except AudioError as e:
            raise HTTPException(422, str(e)) from e


@app.post("/api/analyze/link")
async def analyze_link(url: str = Form(...), prefer: str = Form("auto")):
    if prefer not in ("auto", "youtube", "preview"):
        raise HTTPException(400, "Ungültige Quelle")
    tmp = tempfile.mkdtemp()
    try:
        resolved = await run_in_threadpool(resolve_link, url, Path(tmp), prefer)
        return await run_in_threadpool(_analyze, resolved)
    except (SourceError, AudioError) as e:
        raise HTTPException(422, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
