"""FastAPI HTTP wrapper around Reader.

Loads config from `configs/baseline.yaml` (or `--config` override) on startup.
Defaults to ExtractiveGenerator (no model download needed). Set the
environment variable `RAG_HF_MODEL` to use the Hugging Face pipeline instead.

Includes a simple web UI at `/` with drag-and-drop file upload and question search.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import Config, load_default_config
from src.features import Embedder
from src.generator import ExtractiveGenerator, Generator
from src.reader import Reader

app = FastAPI(title="Mini-RAG Reader", version="0.1.0")

# Lazily initialised; replaced by .load().
_reader: Reader | None = None
_config: Config | None = None
# Store uploaded files in a temp directory
_upload_dir = Path(os.environ.get("MINI_RAG_UPLOAD_DIR", str(Path.home() / ".mini-rag-uploads")))
_upload_dir.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    chunk_id: str
    page_number: int | None
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_default_config()
    return _config


def _build_reader() -> Reader:
    """Build a Reader using the loaded config. Defaults to ExtractiveGenerator.

    Falls back to ExtractiveGenerator if the Hugging Face model
    cannot be loaded (e.g. no network, gated repo, or out of memory).
    """
    config = _get_config()
    embedder = Embedder(model_name=config.embedding.model, device=config.embedding.device)
    generator: Generator
    hf_model = os.environ.get("RAG_HF_MODEL", "")
    if hf_model:
        try:
            generator = Generator(hf_model)
        except Exception:
            generator = ExtractiveGenerator()
    else:
        generator = ExtractiveGenerator()
    return Reader(
        embedder=embedder,
        generator=generator,
        top_k=config.retrieval.top_k,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "reader_loaded": _reader is not None}


@app.get("/")
def index() -> HTMLResponse:
    """Serve the web UI."""
    return HTMLResponse(content=_HTML_PAGE, status_code=200)


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Accept an uploaded PDF or text file and load it into the Reader."""
    global _reader
    save_path: Path | None = None
    try:
        if _reader is None:
            _reader = _build_reader()
        ext = Path(file.filename).suffix.lower()
        if ext not in (".pdf", ".txt"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use .pdf or .txt")
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Cannot upload an empty file.")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        file_id = str(uuid.uuid4())[:8]
        save_path = _upload_dir / f"{file_id}{ext}"
        save_path.write_bytes(content)
        is_text = (ext == ".txt")
        n = _reader.ingest(str(save_path), is_text=is_text)
        if n == 0:
            raise HTTPException(status_code=400, detail="Document is empty or could not be parsed.")
        return {"status": "loaded", "filename": file.filename, "chunks": n, "file_id": file_id, "message": f"Loaded '{file.filename}' with {n} chunks. Ask a question below."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if save_path is not None and save_path.exists():
            try:
                save_path.unlink()
            except Exception:
                pass

@app.post("/clear")
def clear() -> dict:
    """Reset the reader and clear uploaded files."""
    global _reader
    _reader = None
    try:
        for f in _upload_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
    except Exception:
        pass
    return {"status": "cleared", "message": "Document cleared. Upload a new file to start fresh."}


@app.post("/load")
def load(pdf_path: str) -> dict:
    """Lazy-load a PDF into the Reader."""
    global _reader
    try:
        if _reader is None:
            _reader = _build_reader()
        n = _reader.ingest(pdf_path)
        if n == 0:
            raise HTTPException(status_code=400, detail="Document is empty or could not be parsed.")
        return {"chunks": n}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if _reader is None:
        raise HTTPException(status_code=503, detail="Call /load or /upload first.")
    try:
        answer = _reader.ask(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    top_k = req.top_k or _reader.top_k
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        citations=[
            Citation(
                chunk_id=c.chunk.chunk_id,
                page_number=c.chunk.page_number,
                score=c.score,
                text=c.chunk.text,
            )
            for c in answer.citations
        ],
    )


# -- HTML/JS frontend --

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mini-RAG Reader</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa; color: #1a1a2e; min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a3a5c, #2a5c8a);
    color: white; padding: 24px 32px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  }
  .header h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .header p { font-size: 13px; opacity: 0.85; }
  .container { max-width: 800px; margin: 0 auto; padding: 24px 16px; }

  /* Drop zone */
  .drop-zone {
    border: 2px dashed #2a5c8a; border-radius: 12px; padding: 40px 24px;
    text-align: center; background: white; cursor: pointer;
    transition: all 0.2s ease; margin-bottom: 24px;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: #4caf50; background: #f0faf0; transform: scale(1.01);
  }
  .drop-zone .icon { font-size: 40px; margin-bottom: 8px; }
  .drop-zone h3 { color: #2a5c8a; font-size: 15px; margin-bottom: 4px; }
  .drop-zone p { color: #888; font-size: 12px; }
  .drop-zone input[type="file"] { display: none; }

  /* File info */
  .file-info {
    padding: 12px 16px; background: #e8f4e8; border-radius: 8px;
    margin-bottom: 16px; display: none;
  }
  .file-info.visible { display: block; }
  .file-info span { font-weight: 600; color: #2a5c8a; }

  /* Ask box */
  .ask-box {
    display: flex; gap: 8px; margin-bottom: 24px;
  }
  .ask-box input {
    flex: 1; padding: 12px 16px; border: 2px solid #ddd; border-radius: 8px;
    font-size: 14px; outline: none; transition: border-color 0.2s;
  }
  .ask-box input:focus { border-color: #2a5c8a; }
  .ask-box button {
    padding: 12px 24px; background: #2a5c8a; color: white; border: none;
    border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer;
    transition: background 0.2s;
  }
  .ask-box button:hover { background: #1a3a5c; }
  .ask-box button:disabled { background: #ccc; cursor: not-allowed; }

  /* Answer */
  .answer-box {
    background: white; border-radius: 12px; padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); display: none;
  }
  .answer-box.visible { display: block; }
  .answer-box h3 { color: #2a5c8a; margin-bottom: 12px; }
  .answer-box p { line-height: 1.7; margin-bottom: 12px; }
  .answer-box .citations { font-size: 12px; color: #888; }

  /* Status */
  .status { text-align: center; padding: 12px; color: #888; font-size: 13px; }
  .status.error { color: #d32f2f; }
</style>
</head>
<body>
<div class="header">
  <h1>Mini-RAG Reader</h1>
  <p>Upload a PDF or text file and ask questions</p>
</div>
<div class="container">
  <div class="drop-zone" id="dropZone">
    <div class="icon">📄</div>
    <h3>Drop a PDF or TXT file here</h3>
    <p>or click to browse (max 10MB)</p>
    <input type="file" id="fileInput" accept=".pdf,.txt">
  </div>
  <div class="file-info" id="fileInfo"></div>
  <div class="ask-box">
    <input type="text" id="questionInput" placeholder="Ask a question..." disabled>
    <button id="askBtn" disabled>Ask</button>
  </div>
  <div class="status" id="status"></div>
  <div class="answer-box" id="answerBox">
    <h3>Answer</h3>
    <p id="answerText"></p>
    <div class="citations" id="citations"></div>
  </div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const status = document.getElementById('status');
const answerBox = document.getElementById('answerBox');
const answerText = document.getElementById('answerText');
const citations = document.getElementById('citations');

let currentFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

async function handleFile(file) {
  if (file.size > 10 * 1024 * 1024) { showStatus('File too large (max 10MB)', true); return; }
  currentFile = file;
  fileInfo.innerHTML = `<span>${file.name}</span> (${(file.size / 1024).toFixed(1)} KB)`;
  fileInfo.classList.add('visible');
  questionInput.disabled = false;
  askBtn.disabled = false;
  showStatus('');
  answerBox.classList.remove('visible');
}

askBtn.addEventListener('click', askQuestion);
questionInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') askQuestion(); });

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question || !currentFile) return;
  askBtn.disabled = true;
  askBtn.textContent = 'Thinking...';
  showStatus('Processing...');
  answerBox.classList.remove('visible');
  try {
    const formData = new FormData();
    formData.append('file', currentFile);
    const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
    const uploadData = await uploadRes.json();
    if (!uploadData.status) { showStatus(uploadData.detail || 'Upload failed', true); return; }
    const askRes = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });
    const data = await askRes.json();
    answerText.textContent = data.answer;
    citations.textContent = data.citations.map(c => c.chunk_id).join(', ');
    answerBox.classList.add('visible');
    showStatus('');
  } catch (e) {
    showStatus('Error: ' + e.message, true);
  }
  askBtn.disabled = false;
  askBtn.textContent = 'Ask';
}

function showStatus(msg, isError) {
  status.textContent = msg;
  status.className = 'status' + (isError ? ' error' : '');
}
</script>
</body>
</html>"""