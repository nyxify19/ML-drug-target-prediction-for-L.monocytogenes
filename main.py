"""
main.py
───────
FastAPI web service for the L. monocytogenes drug-target prediction pipeline.

Endpoints
─────────
POST /run-pipeline             – submit pipeline as REST background task
GET  /status/{job_id}          – poll job status / results
GET  /results/{job_id}         – download CSV
WS   /ws/pipeline              – WebSocket: streams live logs + final JSON result
GET  /health                   – liveness probe
GET  /docs                     – Swagger UI

Windows note
────────────
Always start with:  python main.py
Never with:         uvicorn main:app --reload  (breaks multiprocessing on Windows)
"""

import asyncio
import json
import logging
import multiprocessing
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────────────
_mp_manager = None   # multiprocessing.Manager() instance, set in lifespan
_executor: Optional[ProcessPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the process pool and MP manager once at startup; clean up on shutdown."""
    global _mp_manager, _executor
    _mp_manager = multiprocessing.Manager()
    _executor   = ProcessPoolExecutor(max_workers=2)
    log.info("Process pool and MP manager started.")
    yield
    _executor.shutdown(wait=False)
    _mp_manager.shutdown()
    log.info("Process pool shut down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="L. monocytogenes Drug-Target Prediction API",
    description=(
        "Run the complete drug-target prediction pipeline. "
        "Use WS /ws/pipeline for live-streamed logs, or "
        "POST /run-pipeline for the classic REST interface."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html and static assets from the same directory
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=_HERE), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(_os.path.join(_HERE, "index.html"))

# ── Job store ─────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class JobRecord(BaseModel):
    job_id:       str
    status:       JobStatus
    submitted_at: float
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    result:       Optional[dict]  = None
    error:        Optional[str]   = None
    output_csv:   Optional[str]   = None


JOBS: dict[str, JobRecord] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────
class RunPipelineRequest(BaseModel):
    output_csv: str = "listeria_final_results.csv"


class JobSubmittedResponse(BaseModel):
    job_id:  str
    message: str
    status:  JobStatus


# ── Helper: run pipeline in a worker process, forwarding queue → async queue ──

def _worker(mp_queue, output_csv: str) -> dict:
    """Runs in ProcessPoolExecutor; imports are local to the worker."""
    from pipeline_core import run_pipeline_streaming  # noqa
    return run_pipeline_streaming(mp_queue, output_csv=output_csv)


async def _drain_mp_queue_to_async(mp_queue, async_queue: asyncio.Queue) -> None:
    """
    Polls the multiprocessing.Queue in a thread-executor so it doesn't
    block the event loop, and forwards each item into an asyncio.Queue.
    """
    import queue as _queue_mod
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(None, mp_queue.get, True, 0.15)
            await async_queue.put(item)
            if item in ("__DONE__", "__ERROR__"):
                break
        except _queue_mod.Empty:
            await asyncio.sleep(0.05)
        except Exception:
            await asyncio.sleep(0.05)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    """
    WebSocket endpoint that:
      1. Runs the full pipeline inside a ProcessPoolExecutor worker.
      2. Streams every log line to the browser as JSON  {"type":"log","data":"..."}
      3. On completion sends                            {"type":"result","data":{...}}
      4. On failure sends                               {"type":"error","data":"..."}
    """
    await websocket.accept()
    log.info("WebSocket connected.")

    # Read optional config from the first message (or use defaults)
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=3.0)
        cfg = json.loads(raw)
    except (asyncio.TimeoutError, Exception):
        cfg = {}

    output_csv = cfg.get("output_csv", "listeria_final_results.csv")

    await websocket.send_text(json.dumps({
        "type": "log",
        "data": "═══ Pipeline starting … ═══",
    }))

    mp_queue   = _mp_manager.Queue()          # reuse the shared Manager
    async_q: asyncio.Queue = asyncio.Queue()
    loop       = asyncio.get_event_loop()

    # Submit the heavy work to the process pool
    future = loop.run_in_executor(_executor, _worker, mp_queue, output_csv)

    # Drain mp_queue → async_q concurrently
    drain_task = asyncio.create_task(_drain_mp_queue_to_async(mp_queue, async_q))

    result: Optional[dict] = None
    error_msg: Optional[str] = None

    try:
        while True:
            try:
                item: str = await asyncio.wait_for(async_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if the future is already done (error path)
                if future.done():
                    break
                continue

            if item == "__DONE__":
                break
            if item == "__ERROR__":
                error_msg = "Pipeline raised an exception (see log above)."
                break

            await websocket.send_text(json.dumps({"type": "log", "data": item}))

        # Await the future to get the return value (or propagate exceptions)
        if error_msg is None:
            try:
                result = await future
            except Exception as exc:
                error_msg = str(exc)

    except WebSocketDisconnect:
        log.warning("WebSocket client disconnected mid-run.")
        drain_task.cancel()
        return
    except Exception as exc:
        error_msg = str(exc)
        log.exception("WebSocket pipeline error: %s", exc)
    finally:
        drain_task.cancel()

    if error_msg:
        await websocket.send_text(json.dumps({"type": "error", "data": error_msg}))
    else:
        await websocket.send_text(json.dumps({
            "type": "log",
            "data": (
                f"═══ Pipeline complete! {result['total_proteins']} proteins | "
                f"AUC {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f} ═══"
            ),
        }))
        await websocket.send_text(json.dumps({"type": "result", "data": result}))

    log.info("WebSocket session closed.")
    try:
        await websocket.close()
    except Exception:
        pass


# ── REST endpoints (unchanged from v1) ───────────────────────────────────────

def _run_pipeline_in_process(job_id: str, output_csv: str) -> dict:
    from pipeline_core import run_full_pipeline  # noqa
    return run_full_pipeline(output_csv=output_csv)


async def _background_job(job_id: str, output_csv: str) -> None:
    JOBS[job_id].status     = JobStatus.RUNNING
    JOBS[job_id].started_at = time.time()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_pipeline_in_process, job_id, output_csv
        )
        JOBS[job_id].status     = JobStatus.DONE
        JOBS[job_id].result     = result
        JOBS[job_id].output_csv = result.get("output_csv")
    except Exception as exc:
        JOBS[job_id].status = JobStatus.FAILED
        JOBS[job_id].error  = str(exc)
        log.exception("Job %s failed: %s", job_id, exc)
    finally:
        JOBS[job_id].finished_at = time.time()


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "active_jobs": sum(
        1 for j in JOBS.values() if j.status == JobStatus.RUNNING
    )}


@app.post("/run-pipeline", response_model=JobSubmittedResponse, status_code=202, tags=["Pipeline"])
async def run_pipeline(body: RunPipelineRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobRecord(job_id=job_id, status=JobStatus.PENDING, submitted_at=time.time())
    background_tasks.add_task(_background_job, job_id, body.output_csv)
    return JobSubmittedResponse(
        job_id=job_id,
        message="Pipeline queued. Poll /status/{job_id} for updates.",
        status=JobStatus.PENDING,
    )


@app.get("/status/{job_id}", response_model=JobRecord, tags=["Pipeline"])
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/results/{job_id}", tags=["Pipeline"])
async def download_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is '{job.status}', not done yet.")
    csv_path = Path(job.output_csv or "listeria_final_results.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Result CSV not found on disk.")
    return FileResponse(path=str(csv_path), media_type="text/csv", filename=csv_path.name)


@app.get("/jobs", tags=["System"])
async def list_jobs():
    return [
        {"job_id": j.job_id, "status": j.status,
         "submitted_at": j.submitted_at, "finished_at": j.finished_at}
        for j in JOBS.values()
    ]


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # The `if __name__` guard is REQUIRED on Windows so that spawned
    # worker processes do not re-execute this block and fork infinitely.
    multiprocessing.freeze_support()   # needed for PyInstaller; harmless otherwise
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
