"""Portable capture bundles and revision forks using the original color bytes."""

import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import replace

from .imaging import SOURCE_LIMIT, frame_input
from .models import (
    MAX_FRAMES, Capacity, RunSettings, Unprocessable, capture_name, choice, flag,
    number, object_value, short_text, validate_keys,
)
from .reports import add_zip_data, freeze_zip_meta, json_bytes

SCHEMA = "focalbench/capture-stack-1"
ARCHIVE_LIMIT = 96 * 1024 * 1024
ARCHIVE_MANIFEST_LIMIT = 192 * 1024
ARCHIVE_DIGEST = re.compile(r"[a-f0-9]{64}")
MANIFEST = "stack.json"


def _active(project: dict) -> list[dict]:
    return [capture for capture in project["captures"] if not capture["archived"]]


def _recipe(project: dict, captures: list[dict]) -> tuple[dict, int | None]:
    settings = dict(project["settings"])
    reference_id = settings.pop("reference_id", None)
    reference_index = next((index for index, capture in enumerate(captures, 1)
                            if capture["id"] == reference_id), None)
    return settings, reference_index


def _asset_name(digest: str) -> str:
    return f"sources/{digest}.capture"


def _portable_frame(frame: dict) -> dict:
    return {name: frame[name] for name in ("name", "position", "included", "note", "sha", "bytes")}


def portable_manifest(project: dict, captures: list[dict]) -> dict:
    config, reference = _recipe(project, captures)
    return {
        "schema": SCHEMA,
        "name": project["name"],
        "note": project["note"],
        "demo": project["demo"],
        "recipe": config,
        "reference_slot": reference,
        "captures": [_portable_frame(frame) for frame in captures],
    }


def _ensure_total(frames: list[dict]) -> dict:
    sources = {}
    for frame in frames:
        digest = frame["sha"]
        size = frame["bytes"]
        if digest in sources and sources[digest] != size:
            raise Unprocessable("A single capture hash is listed with conflicting sizes.")
        sources[digest] = size
    if sum(sources.values()) > ARCHIVE_LIMIT:
        raise Capacity("This stack exceeds the 96 MiB portable source budget.")
    return sources


def portable_export(library, project_id: str) -> tuple[io.IOBase, str]:
    project = library.view_project(project_id)
    captures = _active(project)
    manifest = portable_manifest(project, captures)
    if not captures:
        raise Unprocessable("Add captures before exporting a portable stack.")
    assets = _ensure_total(captures)
    output = tempfile.SpooledTemporaryFile(max_size=12 * 1024 * 1024, mode="w+b")
    try:
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
            add_zip_data(archive, MANIFEST, json_bytes(manifest))
            for digest in sorted(assets):
                entry = freeze_zip_meta(archive, _asset_name(digest))
                path = library.source_path(digest)
                if not path.is_file() or path.stat().st_size != assets[digest]:
                    raise Unprocessable("A referenced source capture is missing or incomplete.")
                with archive.open(entry, "w") as destination, path.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=512 * 1024)
    except Exception:
        output.close()
        raise
    output.seek(0)
    return output, project["name"]


def _load_record(value) -> dict:
    row = object_value(value, "capture")
    validate_keys(row, {"name", "position", "included", "note", "sha", "bytes"}, "capture")
    digest = row.get("sha")
    if not isinstance(digest, str) or ARCHIVE_DIGEST.fullmatch(digest) is None:
        raise Unprocessable("The bundle contains an invalid capture hash.")
    size = row.get("bytes")
    if type(size) is not int or not 1 <= size <= SOURCE_LIMIT:
        raise Capacity("A bundled capture exceeds the supported source size.")
    return {
        "sha": digest,
        "bytes": size,
        "name": capture_name(row.get("name")),
        "position": number(row.get("position"), "focus position", -1e7, 1e7),
        "included": flag(row.get("included"), "included"),
        "note": short_text(row.get("note", ""), "note", maximum=600, allow_blank=True),
    }


def _load_manifest(archive: zipfile.ZipFile) -> dict:
    entries = archive.infolist()
    by_name = {entry.filename: entry for entry in entries}
    if len(by_name) != len(entries) or len(entries) > MAX_FRAMES + 1 or MANIFEST not in by_name:
        raise Unprocessable("The ZIP must contain one FocalBench manifest and its unique source files.")
    if by_name[MANIFEST].file_size > ARCHIVE_MANIFEST_LIMIT:
        raise Capacity("The capture manifest exceeds its limit.")
    try:
        raw = json.loads(archive.read(MANIFEST))
    except (ValueError, UnicodeError) as error:
        raise Unprocessable("The archive manifest cannot be read.") from error
    value = object_value(raw, "manifest")
    if value.get("schema") != SCHEMA:
        raise Unprocessable("Choose a FocalBench portable capture ZIP.")
    validate_keys(value, {"schema", "name", "note", "demo", "recipe", "reference_slot", "captures"})
    frames = value.get("captures")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise Unprocessable(f"A portable stack needs 1–{MAX_FRAMES} captures.")
    frames = [_load_record(frame) for frame in frames]
    sizes = _ensure_total(frames)
    allowed = {MANIFEST, *(_asset_name(digest) for digest in sizes)}
    if allowed != set(by_name):
        raise Unprocessable("The capture ZIP contents do not match the manifest.")
    for digest, size in sizes.items():
        if by_name[_asset_name(digest)].file_size != size:
            raise Unprocessable("A source capture size disagrees with the manifest.")
    reference = value.get("reference_slot")
    if reference is not None:
        if type(reference) is not int or not 1 <= reference <= len(frames):
            raise Unprocessable("The saved reference does not identify a bundle capture.")
        if not frames[reference - 1]["included"]:
            raise Unprocessable("The saved reference is excluded from the bundle.")
    settings = RunSettings.parse(value.get("recipe", {}))
    if settings.reference_id is not None:
        raise Unprocessable("The portable recipe needs a local reference slot.")
    return {
        "name": short_text(value.get("name"), "stack name"),
        "note": short_text(value.get("note", ""), "stack note", maximum=1200, allow_blank=True),
        "demo": None if value.get("demo") is None else choice(value["demo"], "demo", ("circuit", "shim", "plaque")),
        "settings": settings,
        "reference_slot": reference,
        "frames": frames,
    }


def portable_import(library, file, title: str | None = None) -> dict:
    try:
        with zipfile.ZipFile(file) as archive:
            manifest = _load_manifest(archive)
            records = []
            decoded = {}
            for row in manifest["frames"]:
                key = row["sha"]
                if key not in decoded:
                    raw = archive.read(_asset_name(key))
                    if hashlib.sha256(raw).hexdigest() != key:
                        raise Unprocessable(f"The source bytes do not match the recorded hash for {row['name']}.")
                    decoded[key] = frame_input(raw, row["name"], row["position"], library)
                records.append({**decoded[key], **row})
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, OSError) as error:
        raise Unprocessable("The selected ZIP is corrupt, encrypted or unsupported.") from error
    return library.new_project(
        title or manifest["name"], manifest["settings"], manifest["demo"], records,
        reference_slot=manifest["reference_slot"], note=manifest["note"],
    )


def fork_stack(library, project: dict, title: str) -> dict:
    captures = _active(project)
    settings, reference = _recipe(project, captures)
    return library.new_project(
        title, RunSettings.parse(settings), project["demo"], captures,
        reference_slot=reference, note=project["note"],
    )


def fork_result(library, run: dict, title: str) -> dict:
    snapshot = run["snapshot"]
    stored = sorted(snapshot["captures"], key=lambda capture: capture["slot"])
    config = RunSettings.parse(snapshot["settings"])
    reference_id = run["summary"]["reference_id"]
    reference = next((index for index, capture in enumerate(stored, 1)
                      if capture["id"] == reference_id), None)
    return library.new_project(
        title, replace(config, reference_id=None), None, stored,
        reference_slot=reference, note=snapshot.get("project_note", ""),
    )
