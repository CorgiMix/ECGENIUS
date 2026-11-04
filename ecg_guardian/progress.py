"""Writes results/progress.json during a run; the 'ecg-run' Claude Code mod draws it live."""

import json
import os
import time
from pathlib import Path

from .config import RESULTS_DIR

PROGRESS_FILE = RESULTS_DIR / os.environ.get("ECG_PROGRESS_FILE", "progress.json")


class RunProgress:
    def __init__(self, run: str, n_records: int, n_af: int, stages: list[tuple[str, int]]):
        self.data = {"run": run, "n_records": n_records, "n_af": n_af,
                     "started_at": int(time.time() * 1000), "updated_at": int(time.time() * 1000),
                     "finished": False, "results": [], "report": None, "error": None,
                     "stages": [{"name": n, "done": 0, "total": t, "status": "waiting"} for n, t in stages]}
        self._last_write = 0.0
        self.write(force=True)

    def _stage(self, name):
        return next(s for s in self.data["stages"] if s["name"] == name)

    def start(self, name):
        self._stage(name)["status"] = "running"; self.write(force=True)

    def advance(self, name, n=1):
        s = self._stage(name); s["done"] = min(s["total"], s["done"] + n); self.write()

    def finish(self, name):
        s = self._stage(name); s["done"] = s["total"]; s["status"] = "done"; self.write(force=True)

    def done(self, results: list[dict], report: str | None):
        self.data.update(finished=True, results=results, report=report); self.write(force=True)

    def fail(self, msg: str):
        self.data.update(finished=True, error=msg); self.write(force=True)

    def write(self, force=False):
        now = time.time()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        self.data["updated_at"] = int(now * 1000)
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        # Progress is cosmetic: it must never crash the run. On Windows the swap fails while the
        # pane (or OneDrive) has the file open, so retry briefly, then skip this update.
        for _ in range(5):
            try:
                tmp.write_text(json.dumps(self.data), encoding="utf-8")
                os.replace(tmp, PROGRESS_FILE)    # atomic: the pane never reads a half-written file
                return
            except OSError:
                time.sleep(0.05)
