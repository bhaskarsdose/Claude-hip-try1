"""In-memory job tracker for the reconstruction web app."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus = "pending"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    used_trained_model: bool = False
    workdir: Path | None = None

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "used_trained_model": self.used_trained_model,
        }


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, workdir: Path) -> Job:
        async with self._lock:
            jid = uuid.uuid4().hex
            job = Job(id=jid, workdir=workdir)
            self._jobs[jid] = job
            return job

    async def get(self, jid: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(jid)

    async def update(self, jid: str, **fields) -> None:
        async with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            if fields.get("status") in ("done", "failed"):
                job.finished_at = time.time()
