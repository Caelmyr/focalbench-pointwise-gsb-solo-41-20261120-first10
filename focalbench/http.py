"""Shared HTTP request and streamed artifact helpers."""

import io

from flask import current_app, request, send_file

from .models import Capacity, Region, WorkbenchError, integer, object_value
from .reports import clean_filename, sheet_bytes


def library():
    return current_app.extensions["focalbench"]["library"]


def engine():
    return current_app.extensions["focalbench"]["engine"]


def payload():
    return object_value(request.get_json())


def optional_payload():
    if not request.data:
        return {}
    return payload()


def revision(raw):
    value = raw.get("revision")
    return None if value in (None, "") else integer(value, "revision", 1, 1_000_000_000)


def bytes_file(stream, limit: int):
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise Capacity(f"Selected file exceeds {limit / 1024 / 1024:.1f} MiB.")
    return data


def one_upload(label: str):
    stream = request.files.get(label)
    if stream is None or not stream.filename:
        raise WorkbenchError("Choose a file first.", field=label)
    return stream


def png_response(in_memory, heading: str | None = None):
    buf = io.BytesIO(sheet_bytes(in_memory))
    return file_response(buf, "image/png", heading)


def file_response(file, mime: str, attachment: str | None = None):
    if isinstance(file, bytes):
        file = io.BytesIO(file)
    return send_file(
        file, mimetype=mime, as_attachment=attachment is not None,
        download_name=attachment, max_age=0,
    )


def file_title(title: str, suffix: str):
    return clean_filename(title) + suffix


def region_of(values: dict, name: str = "region"):
    raw = values.get(name)
    return None if raw is None else Region.parse(raw)


def region_query():
    keys = ("left", "top", "right", "bottom")
    if not any(key in request.args for key in keys):
        return None
    return Region.parse({key: request.args.get(key) for key in keys})
