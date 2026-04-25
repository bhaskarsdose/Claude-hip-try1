"""FastAPI server for the hip-reconstruction web UI.

Endpoints
---------
POST /api/reconstruct      : multipart upload of a .nii.gz mask -> {job_id}
GET  /api/jobs/{id}        : status JSON
GET  /api/jobs/{id}/input.stl
GET  /api/jobs/{id}/implant.stl
GET  /                     : static UI (index.html + app.js + three.js CDN)
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import OUTPUTS_DIR, ensure_dirs
from ..infer import reconstruct_from_nifti
from .jobs import JobStore

ALLOWED_SUFFIXES = (".nii", ".nii.gz")

app = FastAPI(title="Hip Bone Reconstruction")
jobs = JobStore()
_static_dir = Path(__file__).parent / "static"


@app.on_event("startup")
async def _startup() -> None:
    ensure_dirs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _has_nifti_suffix(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def _run_reconstruction(job_id: str, input_path: Path, workdir: Path) -> None:
    """Synchronous worker scheduled via BackgroundTasks. Updates the JobStore."""
    import asyncio

    async def _set(**fields):
        await jobs.update(job_id, **fields)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_set(status="running"))
        result = reconstruct_from_nifti(input_path)

        input_stl = workdir / "input.stl"
        implant_stl = workdir / "implant.stl"
        if len(result.input_mesh.vertices) > 0:
            result.input_mesh.export(str(input_stl))
        else:
            input_stl.write_bytes(b"")  # empty placeholder so download endpoint exists
        if len(result.implant_mesh.vertices) > 0:
            result.implant_mesh.export(str(implant_stl))
        else:
            implant_stl.write_bytes(b"")

        loop.run_until_complete(
            _set(status="done", used_trained_model=result.used_trained_model)
        )
    except Exception as exc:
        traceback.print_exc()
        loop.run_until_complete(_set(status="failed", error=str(exc)))
    finally:
        loop.close()


@app.post("/api/reconstruct")
async def reconstruct(background: BackgroundTasks, file: UploadFile = File(...)):
    if not _has_nifti_suffix(file.filename or ""):
        raise HTTPException(400, f"Expected .nii or .nii.gz, got: {file.filename}")

    workdir = Path(tempfile.mkdtemp(prefix="hiprec_", dir=str(OUTPUTS_DIR)))
    in_name = "input.nii.gz" if (file.filename or "").lower().endswith(".gz") else "input.nii"
    in_path = workdir / in_name
    with in_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = await jobs.create(workdir)
    background.add_task(_run_reconstruction, job.id, in_path, workdir)
    return {"job_id": job.id}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    job = await jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.to_public()


def _serve_artifact(workdir: Path | None, name: str) -> FileResponse:
    if workdir is None:
        raise HTTPException(404, "job has no workdir")
    p = workdir / name
    if not p.is_file():
        raise HTTPException(404, f"{name} not ready")
    return FileResponse(str(p), media_type="model/stl", filename=name)


@app.get("/api/jobs/{jid}/input.stl")
async def get_input_stl(jid: str):
    job = await jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return _serve_artifact(job.workdir, "input.stl")


@app.get("/api/jobs/{jid}/implant.stl")
async def get_implant_stl(jid: str):
    job = await jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    return _serve_artifact(job.workdir, "implant.stl")


# Static UI must be mounted last so the API routes above take priority.
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
