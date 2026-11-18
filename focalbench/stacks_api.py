"""Capture library, metadata, preflight, and planning routes."""

from flask import Blueprint, request, send_file

from .acquisition import inspect_schedule
from .demos import DEMOS, create_demo, demo_catalog
from .http import (
    bytes_file, file_response, file_title, library, one_upload, optional_payload,
    payload, png_response, region_of, region_query, revision,
)
from .imaging import SOURCE_LIMIT, frame_input
from .models import (
    DEFAULT_NAME, MAX_FRAMES, CapturePlan, Capacity, Conflict, FrameEdit, RunSettings,
    WorkbenchError, choice, flag, integer, number, short_text, validate_keys,
)
from .quality import inspect_stack
from .radiometry import capture_histogram, exposure_map
from .reports import csv_bytes, inspection_sheet, schedule_report
from .staging import MAX_LOG_BYTES, stage_log
from .transfer import ARCHIVE_LIMIT, fork_stack, portable_export, portable_import

stacks = Blueprint("stacks", __name__, url_prefix="/api")
MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff", "WEBP": "image/webp"}


@stacks.get("/demos")
def list_demos():
    return {"demos": demo_catalog()}


@stacks.get("/stacks")
def list_stacks():
    return {"stacks": library().list_projects()}


@stacks.post("/stacks")
def new_stack():
    data = optional_payload()
    validate_keys(data, {"name", "settings"})
    name = short_text(data.get("name", DEFAULT_NAME), "stack name")
    return library().new_project(name, RunSettings.parse(data.get("settings", {}))), 201


@stacks.post("/stacks/demo")
def demo_stack():
    data = payload()
    validate_keys(data, {"name", "demo"})
    demo = choice(data.get("demo"), "demo", tuple(DEMOS))
    default = DEMOS[demo].label
    title = short_text(data.get("name", default), "stack name")
    return create_demo(library(), demo, title), 201


@stacks.post("/stacks/import")
def import_bundle():
    archive = one_upload("archive")
    archive.stream.seek(0, 2)
    size = archive.stream.tell()
    archive.stream.seek(0)
    if size > ARCHIVE_LIMIT + 1_000_000:
        raise Capacity("Upload a portable ZIP up to 97 MiB.")
    name = request.form.get("name")
    title = short_text(name, "stack name") if name else None
    return portable_import(library(), archive.stream, title), 201


@stacks.get("/stacks/<project_id>")
def fetch_stack(project_id):
    return library().view_project(project_id)


@stacks.patch("/stacks/<project_id>")
def change_stack(project_id):
    data = payload()
    validate_keys(data, {"name", "note", "settings", "revision"})
    updated = {}
    if "name" in data:
        updated["name"] = short_text(data["name"], "stack name")
    if "note" in data:
        updated["note"] = short_text(data["note"], "stack note", maximum=1200, allow_blank=True)
    if "settings" in data:
        updated["settings"] = RunSettings.parse(data["settings"])
    if not updated:
        raise WorkbenchError("Choose a stack field to update.")
    return library().update_project(project_id, updated, revision(data))


@stacks.get("/stacks/<project_id>/portable")
def bundle_stack(project_id):
    content, name = portable_export(library(), project_id)
    return file_response(content, "application/zip", file_title(name, "-capture-stack.zip"))


@stacks.post("/stacks/<project_id>/fork")
def duplicate_stack(project_id):
    project = library().view_project(project_id)
    data = payload()
    validate_keys(data, {"name"})
    title = short_text(data.get("name"), "stack name")
    return fork_stack(library(), project, title), 201


@stacks.post("/stacks/<project_id>/captures")
def upload_captures(project_id):
    project = library().view_project(project_id)
    sources = request.files.getlist("images")
    if not sources or not all(item.filename for item in sources):
        raise WorkbenchError("Choose one or more capture images.", field="images")
    if len(sources) > 16:
        raise Capacity("Upload up to sixteen images in one batch.")
    if len(project["captures"]) + len(sources) > MAX_FRAMES:
        raise Capacity(f"The stack has space for {MAX_FRAMES - len(project['captures'])} more captures.")
    rail_positions = request.form.getlist("positions")
    if rail_positions and len(rail_positions) != len(sources):
        raise WorkbenchError("Supply one rail position per image.", field="positions")
    start = max((item["position"] for item in project["captures"] if not item["archived"]), default=-1) + 1
    values = [
        number(value, "focus position", -1e7, 1e7)
        for value in (rail_positions or [start + index for index in range(len(sources))])
    ]
    expected = revision(request.form) or project["revision"]
    records = [
        frame_input(bytes_file(stream, SOURCE_LIMIT), stream.filename, rail, library())
        for stream, rail in zip(sources, values)
    ]
    return library().append_captures(project_id, records, expected), 201


@stacks.patch("/stacks/<project_id>/captures/<capture_id>")
def change_capture(project_id, capture_id):
    data = payload()
    expected = revision(data)
    fields = {key: value for key, value in data.items() if key != "revision"}
    return library().edit_capture(project_id, capture_id, FrameEdit.parse(fields), expected)


@stacks.post("/stacks/<project_id>/captures/<capture_id>/retake")
def replace_capture(project_id, capture_id):
    old = library().capture(project_id, capture_id)
    stream = one_upload("image")
    source = bytes_file(stream, SOURCE_LIMIT)
    captured = frame_input(source, stream.filename, old["position"], library())
    return library().retake(project_id, capture_id, captured, revision(request.form)), 201


@stacks.get("/stacks/<project_id>/captures/<capture_id>/preview")
def preview_capture(project_id, capture_id):
    capture = library().capture(project_id, capture_id)
    return send_file(library().preview_path(capture["sha"]), mimetype="image/webp", max_age=31_536_000)


@stacks.get("/stacks/<project_id>/captures/<capture_id>/source")
def get_source(project_id, capture_id):
    capture = library().capture(project_id, capture_id)
    mime = MIME[capture["information"]["format"]]
    return send_file(
        library().source_path(capture["sha"]), mimetype=mime, max_age=0,
        as_attachment=True, download_name=capture["name"],
    )


@stacks.get("/stacks/<project_id>/preflight")
def preflight_stack(project_id):
    project = library().view_project(project_id)
    preflight = inspect_stack(library(), project["captures"])
    return {"revision": project["revision"], **preflight}


@stacks.post("/stacks/<project_id>/captures/<capture_id>/radiometry")
def capture_radiometry(project_id, capture_id):
    capture = library().capture(project_id, capture_id)
    data = optional_payload()
    validate_keys(data, {"region"})
    return capture_histogram(library(), capture, region_of(data))


@stacks.get("/stacks/<project_id>/captures/<capture_id>/radiometry.png")
def view_exposure(project_id, capture_id):
    capture = library().capture(project_id, capture_id)
    mode = choice(request.args.get("mode", "both"), "clipping map", ("clipped", "shadow", "both"))
    return png_response(exposure_map(library(), capture, mode, region_query()))


@stacks.get("/stacks/<project_id>/contact.png")
def contact_print(project_id):
    project = library().view_project(project_id)
    columns = integer(request.args.get("columns", 3), "columns", 2, 5)
    archived = request.args.get("archived", "0") == "1"
    captures = [item for item in project["captures"] if archived or not item["archived"]]
    rail_unit = project["settings"]["scale"]["rail_unit"]
    title = project["name"]
    sheet = inspection_sheet(library(), captures, title, rail_unit, columns)
    heading = file_title(title, "-capture-proof.png") if request.args.get("download") else None
    return png_response(sheet, heading)


@stacks.post("/stacks/<project_id>/positions-log")
def apply_positions_log(project_id):
    project = library().view_project(project_id)
    upload = one_upload("log")
    mode = choice(request.form.get("mode", "preview"), "log mode", ("preview", "apply"))
    result, positions = stage_log(project, bytes_file(upload, MAX_LOG_BYTES))
    if mode == "preview":
        return result
    expected = revision(request.form)
    if expected != project["revision"]:
        raise Conflict("Preview the rail log again against the current stack revision.")
    if not result["valid"]:
        raise Conflict("Resolve unmatched captures and position collisions before applying.", details=result)
    attached = library().apply_positions(project_id, positions, expected)
    return {"report": result, "stack": attached}


@stacks.post("/stacks/<project_id>/schedule")
def schedule_stack(project_id):
    project = library().view_project(project_id)
    data = payload()
    validate_keys(data, {"plan", "download"})
    unit = project["settings"]["scale"]["rail_unit"]
    plan = CapturePlan.parse(data.get("plan"))
    result = inspect_schedule(plan, project["captures"])
    if flag(data.get("download", False), "download"):
        name = file_title(project["name"], "-rail-schedule.csv")
        return file_response(schedule_report(result, unit), "text/csv", name)
    return {"unit": unit, **result}


@stacks.get("/stacks/<project_id>/capture-log.csv")
def current_position_log(project_id):
    stack = library().view_project(project_id)
    unit = stack["settings"]["scale"]["rail_unit"]
    captures = [capture for capture in stack["captures"] if not capture["archived"]]
    slots = [
        (capture["name"], capture["position"], unit, capture["slot"], capture["included"], capture["sha"])
        for capture in captures
    ]
    report = csv_bytes(("filename", "position", "unit", "slot", "included", "source_sha256"), slots)
    return file_response(report, "text/csv", file_title(stack["name"], "-rail-log.csv"))
