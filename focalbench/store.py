"""SQLite metadata and immutable image assets for a local FocalBench library."""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from .models import (
    MAX_FRAMES, MAX_MARKS, RUN_ASSETS, Capacity, Conflict, FrameEdit, Mark,
    Missing, RunSettings, require_id, storage_id, utc_now,
)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    revision INTEGER NOT NULL,
    settings TEXT NOT NULL,
    demo TEXT,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    slot INTEGER NOT NULL,
    position REAL NOT NULL,
    name TEXT NOT NULL,
    sha TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    information TEXT NOT NULL,
    included INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    superseded_by TEXT,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS captures_stack ON captures(project_id, slot, recorded_at);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_stack ON runs(project_id, created_at DESC);
CREATE TABLE IF NOT EXISTS marks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    mark TEXT NOT NULL,
    measurement TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS marks_run ON marks(run_id, created_at);
CREATE TABLE IF NOT EXISTS review_cards (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_cards_run ON review_cards(run_id,saved_at);
"""


def pack(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def project_record(row: sqlite3.Row) -> dict:
    value = dict(row)
    value["settings"] = json.loads(value["settings"])
    return value


def capture_record(row: sqlite3.Row) -> dict:
    value = dict(row)
    value["information"] = json.loads(value["information"])
    value["included"] = bool(value["included"])
    value["archived"] = value["superseded_by"] is not None
    return value


def run_record(row: sqlite3.Row) -> dict:
    value = dict(row)
    value["snapshot"] = json.loads(value["snapshot"])
    value["summary"] = json.loads(value["summary"])
    return value


def card_record(row: sqlite3.Row) -> dict:
    value = dict(row)
    value["content"] = json.loads(value["content"])
    return value


def mark_record(row: sqlite3.Row) -> dict:
    value = dict(row)
    value.update(json.loads(value.pop("mark")))
    value["measurement"] = json.loads(value["measurement"])
    return value


class Library:
    def __init__(self, directory: Path):
        self.root = directory.resolve()
        self.root.mkdir(exist_ok=True, parents=True)
        self.database = self.root / "library.sqlite3"
        self.sources = self.root / "sources"
        self.previews = self.root / "previews"
        self.products = self.root / "runs"
        for folder in (self.sources, self.previews, self.products):
            folder.mkdir(exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(SCHEMA)
            if db.execute("PRAGMA user_version").fetchone()[0] == 0:
                db.execute("PRAGMA user_version = 1")

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, isolation_level=None, timeout=8)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 8000")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def check(self) -> dict:
        with self.connect() as db:
            db.execute("SELECT 1").fetchone()
            version = db.execute("PRAGMA user_version").fetchone()[0]
        return {"ready": True, "schema": version}

    @staticmethod
    def _project(db: sqlite3.Connection, project_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM projects WHERE id = ?",
            (require_id(project_id, "stack"),),
        ).fetchone()
        if row is None:
            raise Missing("Stack does not exist.")
        return row

    @staticmethod
    def _capture(db: sqlite3.Connection, project_id: str, frame_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM captures WHERE project_id = ? AND id = ?",
            (require_id(project_id, "stack"), require_id(frame_id, "capture")),
        ).fetchone()
        if row is None:
            raise Missing("Capture does not exist in this stack.")
        return row

    @staticmethod
    def _run(db: sqlite3.Connection, project_id: str, run_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM runs WHERE project_id = ? AND id = ?",
            (require_id(project_id, "stack"), require_id(run_id, "result")),
        ).fetchone()
        if row is None:
            raise Missing("Result does not exist in this stack.")
        return row

    @staticmethod
    def _unchanged(row: sqlite3.Row, expected_revision: int | None) -> None:
        if expected_revision is not None and row["revision"] != expected_revision:
            raise Conflict(
                "This stack changed. Refresh before applying the edit.",
                details={"revision": row["revision"]},
            )

    @staticmethod
    def _touch(db: sqlite3.Connection, project_id: str) -> None:
        db.execute(
            "UPDATE projects SET revision = revision + 1, changed_at = ? WHERE id = ?",
            (utc_now(), project_id),
        )

    def list_projects(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT p.*,
                    (SELECT COUNT(*) FROM captures c WHERE c.project_id=p.id
                     AND c.superseded_by IS NULL) AS capture_count,
                    (SELECT COUNT(*) FROM runs r WHERE r.project_id=p.id) AS result_count
                FROM projects p ORDER BY p.changed_at DESC, p.id
                """
            ).fetchall()
        return [project_record(row) for row in rows]

    def project(self, project_id: str) -> dict:
        with self.connect() as db:
            return project_record(self._project(db, project_id))

    def view_project(self, project_id: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN")
            project = project_record(self._project(db, project_id))
            project["captures"] = self._captures(db, project_id, include_archived=True)
            project["results"] = self._result_index(db, project_id)
        return project

    def new_project(
        self,
        name: str,
        settings: RunSettings,
        demo: str | None = None,
        captures: list[dict] | None = None,
        *,
        reference_slot: int | None = None,
        note: str = "",
    ) -> dict:
        project_id = storage_id()
        timestamp = utc_now()
        captures = captures or []
        if len(captures) > MAX_FRAMES:
            raise Capacity(f"A stack can retain at most {MAX_FRAMES} captures.")
        with self.connect(write=True) as db:
            db.execute(
                """
                INSERT INTO projects(id,name,created_at,changed_at,revision,settings,demo,note)
                VALUES (?,?,?,?,1,?,?,?)
                """,
                (project_id, name, timestamp, timestamp, pack(settings.public()), demo, note),
            )
            for slot, incoming in enumerate(captures, start=1):
                fresh_id = self._insert_capture(db, project_id, slot, incoming)
                if slot == reference_slot:
                    scoped = replace(settings, reference_id=fresh_id)
                    db.execute("UPDATE projects SET settings=? WHERE id=?", (pack(scoped.public()), project_id))
        return self.view_project(project_id)

    def update_project(
        self,
        project_id: str,
        updates: dict,
        expected_revision: int | None,
    ) -> dict:
        with self.connect(write=True) as db:
            row = self._project(db, project_id)
            self._unchanged(row, expected_revision)
            if "settings" in updates:
                settings = updates["settings"]
                self._check_reference(db, project_id, settings.reference_id)
                db.execute(
                    "UPDATE projects SET settings=? WHERE id=?",
                    (pack(settings.public()), project_id),
                )
            if "name" in updates:
                db.execute("UPDATE projects SET name=? WHERE id=?", (updates["name"], project_id))
            if "note" in updates:
                db.execute("UPDATE projects SET note=? WHERE id=?", (updates["note"], project_id))
            self._touch(db, project_id)
        return self.view_project(project_id)

    def _check_reference(self, db, project_id: str, reference_id: str | None) -> None:
        if reference_id is None:
            return
        reference = self._capture(db, project_id, reference_id)
        if not reference["included"] or reference["superseded_by"]:
            raise Conflict("The reference capture must be included in the current stack.")

    @staticmethod
    def _captures(db: sqlite3.Connection, project_id: str, *, include_archived: bool) -> list[dict]:
        if include_archived:
            rows = db.execute(
                """
                SELECT * FROM captures WHERE project_id=?
                ORDER BY slot, superseded_by IS NOT NULL, recorded_at DESC, id
                """,
                (project_id,),
            )
        else:
            rows = db.execute(
                """
                SELECT * FROM captures WHERE project_id=? AND superseded_by IS NULL
                ORDER BY slot, id
                """,
                (project_id,),
            )
        return [capture_record(row) for row in rows]

    def active_captures(self, project_id: str) -> list[dict]:
        with self.connect() as db:
            return self._captures(db, project_id, include_archived=False)

    def capture(self, project_id: str, frame_id: str) -> dict:
        with self.connect() as db:
            return capture_record(self._capture(db, project_id, frame_id))

    @staticmethod
    def _retained(db: sqlite3.Connection, project_id: str) -> int:
        return db.execute("SELECT COUNT(*) FROM captures WHERE project_id=?", (project_id,)).fetchone()[0]

    @staticmethod
    def _next_slot(db: sqlite3.Connection, project_id: str) -> int:
        return db.execute(
            "SELECT COALESCE(MAX(slot),0)+1 FROM captures WHERE project_id=?",
            (project_id,),
        ).fetchone()[0]

    @staticmethod
    def _insert_capture(
        db: sqlite3.Connection,
        project_id: str,
        slot: int,
        incoming: dict,
    ) -> str:
        frame_id = storage_id()
        db.execute(
            """
            INSERT INTO captures(
                id,project_id,slot,position,name,sha,width,height,bytes,
                information,included,recorded_at,note
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                frame_id, project_id, slot, incoming["position"], incoming["name"],
                incoming["sha"], incoming["width"], incoming["height"], incoming["bytes"],
                pack(incoming["information"]), bool(incoming.get("included", True)),
                utc_now(), incoming.get("note", ""),
            ),
        )
        return frame_id

    def append_captures(
        self,
        project_id: str,
        captures: list[dict],
        expected_revision: int | None,
    ) -> dict:
        with self.connect(write=True) as db:
            stack = self._project(db, project_id)
            self._unchanged(stack, expected_revision)
            if self._retained(db, project_id) + len(captures) > MAX_FRAMES:
                raise Capacity(f"A stack can retain at most {MAX_FRAMES} captures.")
            slot = self._next_slot(db, project_id)
            for offset, incoming in enumerate(captures):
                self._insert_capture(db, project_id, slot + offset, incoming)
            self._touch(db, project_id)
        return self.view_project(project_id)

    def retake(
        self,
        project_id: str,
        frame_id: str,
        incoming: dict,
        expected_revision: int | None,
    ) -> dict:
        with self.connect(write=True) as db:
            stack = self._project(db, project_id)
            self._unchanged(stack, expected_revision)
            current = self._capture(db, project_id, frame_id)
            if current["superseded_by"]:
                raise Conflict("Replace the current capture in the slot.")
            if self._retained(db, project_id) >= MAX_FRAMES:
                raise Capacity(f"A stack can retain at most {MAX_FRAMES} captures.")
            incoming = {
                **incoming,
                "position": current["position"],
                "included": current["included"],
                "note": current["note"],
            }
            replacement_id = self._insert_capture(db, project_id, current["slot"], incoming)
            db.execute(
                "UPDATE captures SET superseded_by=?, included=0 WHERE id=?",
                (replacement_id, frame_id),
            )
            settings = project_record(stack)["settings"]
            if settings.get("reference_id") == frame_id:
                settings["reference_id"] = replacement_id
                db.execute("UPDATE projects SET settings=? WHERE id=?", (pack(settings), project_id))
            self._touch(db, project_id)
        return self.view_project(project_id)

    def edit_capture(
        self,
        project_id: str,
        frame_id: str,
        edit: FrameEdit,
        expected_revision: int | None,
    ) -> dict:
        with self.connect(write=True) as db:
            stack = self._project(db, project_id)
            self._unchanged(stack, expected_revision)
            current = self._capture(db, project_id, frame_id)
            if current["superseded_by"]:
                raise Conflict("Archived captures cannot be edited.")
            db.execute(
                "UPDATE captures SET position=?,included=?,note=? WHERE id=?",
                (
                    current["position"] if edit.position is None else edit.position,
                    current["included"] if edit.enabled is None else edit.enabled,
                    current["note"] if edit.note is None else edit.note,
                    frame_id,
                ),
            )
            if edit.enabled is False:
                settings = project_record(stack)["settings"]
                if settings.get("reference_id") == frame_id:
                    settings["reference_id"] = None
                    db.execute(
                        "UPDATE projects SET settings=? WHERE id=?",
                        (pack(settings), project_id),
                    )
            self._touch(db, project_id)
        return self.view_project(project_id)

    def snapshot(self, project_id: str, expected_revision: int | None = None) -> dict:
        with self.connect() as db:
            db.execute("BEGIN")
            row = self._project(db, project_id)
            self._unchanged(row, expected_revision)
            project = project_record(row)
            captures = self._captures(db, project_id, include_archived=False)
            return {
                "project_id": project["id"],
                "project_name": project["name"],
                "project_note": project["note"],
                "revision": project["revision"],
                "settings": project["settings"],
                "captures": [capture for capture in captures if capture["included"]],
            }

    def _result_index(self, db: sqlite3.Connection, project_id: str) -> list[dict]:
        rows = db.execute(
            """
            SELECT id, title, created_at, source_revision, summary
            FROM runs WHERE project_id=? ORDER BY created_at DESC, id
            """,
            (project_id,),
        ).fetchall()
        index = []
        for row in rows:
            summary = json.loads(row["summary"])
            index.append({
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "source_revision": row["source_revision"],
                "width": summary["width"],
                "height": summary["height"],
                "capture_count": summary["capture_count"],
                "reliable_fraction": summary["reliable_fraction"],
            })
        return index

    def run(self, project_id: str, run_id: str) -> dict:
        with self.connect() as db:
            return run_record(self._run(db, project_id, run_id))

    def publish_run(
        self,
        project_id: str,
        run_id: str,
        title: str,
        snapshot: dict,
        summary: dict,
    ) -> dict:
        with self.connect(write=True) as db:
            stack = self._project(db, project_id)
            self._unchanged(stack, snapshot["revision"])
            if not self.run_directory(run_id).is_dir():
                raise Missing("The computed result files are incomplete.")
            db.execute(
                """
                INSERT INTO runs(id,project_id,title,created_at,source_revision,snapshot,summary)
                VALUES(?,?,?,?,?,?,?)
                """,
                (run_id, project_id, title, utc_now(), snapshot["revision"], pack(snapshot), pack(summary)),
            )
            db.execute("UPDATE projects SET changed_at=? WHERE id=?", (utc_now(), project_id))
            return run_record(self._run(db, project_id, run_id))

    def run_directory(self, run_id: str) -> Path:
        return self.products / require_id(run_id, "result")

    def run_asset(self, project_id: str, run_id: str, name: str) -> Path:
        if name not in RUN_ASSETS:
            raise Missing("Unknown result image.")
        self.run(project_id, run_id)
        result = self.run_directory(run_id) / RUN_ASSETS[name]
        if not result.is_file():
            raise Missing("Result image is missing.")
        return result

    def list_marks(self, project_id: str, run_id: str) -> list[dict]:
        with self.connect() as db:
            self._run(db, project_id, run_id)
            rows = db.execute(
                "SELECT * FROM marks WHERE run_id=? ORDER BY created_at,id",
                (run_id,),
            )
            return [mark_record(row) for row in rows]

    def add_mark(
        self,
        project_id: str,
        run_id: str,
        value: Mark,
        measurement: dict,
    ) -> dict:
        with self.connect(write=True) as db:
            self._run(db, project_id, run_id)
            count = db.execute("SELECT COUNT(*) FROM marks WHERE run_id=?", (run_id,)).fetchone()[0]
            if count >= MAX_MARKS:
                raise Capacity(f"A result can retain at most {MAX_MARKS} annotations.")
            mark_id = storage_id()
            db.execute(
                "INSERT INTO marks(id,run_id,mark,measurement,created_at) VALUES (?,?,?,?,?)",
                (mark_id, run_id, pack(value.public()), pack(measurement), utc_now()),
            )
            row = db.execute("SELECT * FROM marks WHERE id=?", (mark_id,)).fetchone()
            return mark_record(row)

    def remove_mark(self, project_id: str, run_id: str, mark_id: str) -> None:
        with self.connect(write=True) as db:
            self._run(db, project_id, run_id)
            removed = db.execute(
                "DELETE FROM marks WHERE run_id=? AND id=?",
                (run_id, require_id(mark_id, "annotation")),
            )
            if removed.rowcount == 0:
                raise Missing("Annotation does not exist in this result.")

    @staticmethod
    def _hashed_path(directory: Path, digest: str, extension: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Invalid stored image digest")
        return directory / digest[:2] / (digest + extension)

    def source_path(self, digest: str) -> Path:
        return self._hashed_path(self.sources, digest, ".bin")

    def preview_path(self, digest: str) -> Path:
        return self._hashed_path(self.previews, digest, ".webp")

    def store_source(self, source: bytes, preview: bytes) -> str:
        digest = hashlib.sha256(source).hexdigest()
        self._write_blob(self.source_path(digest), source)
        self._write_blob(self.preview_path(digest), preview)
        return digest

    @staticmethod
    def _write_blob(path: Path, contents: bytes) -> None:
        if path.exists():
            return
        path.parent.mkdir(exist_ok=True, parents=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".incoming-", delete=False) as target:
            temp = Path(target.name)
            try:
                target.write(contents)
                target.flush()
                os.fsync(target.fileno())
                target.close()
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)

    def apply_positions(
        self, project_id: str, positions: dict[str, float], expected_revision: int,
    ) -> dict:
        with self.connect(write=True) as db:
            project = self._project(db, project_id)
            self._unchanged(project, expected_revision)
            captures = self._captures(db, project_id, include_archived=False)
            known = {capture["id"] for capture in captures}
            if not set(positions) <= known:
                raise Conflict("One or more captures have been replaced. Refresh the stack.")
            updates = ((value, frame_id) for frame_id, value in positions.items())
            db.executemany("UPDATE captures SET position=? WHERE id=?", updates)
            self._touch(db, project_id)
        return self.view_project(project_id)


    def _card(self, db, run_id: str, card_id: str):
        row = db.execute(
            "SELECT * FROM review_cards WHERE run_id=? AND id=?",
            (run_id, require_id(card_id, "inspection card")),
        ).fetchone()
        if row is None:
            raise Missing("Inspection card does not exist in this result.")
        return row

    def review_cards(self, project_id: str, run_id: str) -> list[dict]:
        with self.connect() as db:
            self._run(db, project_id, run_id)
            rows = db.execute(
                "SELECT * FROM review_cards WHERE run_id=? ORDER BY saved_at DESC,id",
                (run_id,),
            )
            return [card_record(row) for row in rows]

    def review_card(self, project_id: str, run_id: str, card_id: str) -> dict:
        with self.connect() as db:
            self._run(db, project_id, run_id)
            return card_record(self._card(db, run_id, card_id))

    def add_card(self, project_id: str, run_id: str, title: str, content: dict):
        with self.connect(write=True) as db:
            self._run(db, project_id, run_id)
            count = db.execute(
                "SELECT COUNT(*) FROM review_cards WHERE run_id=?", (run_id,),
            ).fetchone()[0]
            if count >= 24:
                raise Capacity("A result can retain up to 24 inspection cards.")
            card_id = storage_id()
            db.execute(
                "INSERT INTO review_cards(id,run_id,title,content,saved_at) VALUES(?,?,?,?,?)",
                (card_id, run_id, title, pack(content), utc_now()),
            )
            return card_record(self._card(db, run_id, card_id))

    def delete_card(self, project_id: str, run_id: str, card_id: str) -> None:
        with self.connect(write=True) as db:
            self._run(db, project_id, run_id)
            self._card(db, run_id, card_id)
            db.execute("DELETE FROM review_cards WHERE id=? AND run_id=?", (card_id, run_id))
