"""Atomic OCR artifact publication and image validation helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from PIL import Image


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish bytes through a unique same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def write_json_atomic(path: Path, payload: object) -> None:
    """Publish a JSON artifact only after serialization succeeds."""
    write_bytes_atomic(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
    )


def image_file_is_valid(path: Path) -> bool:
    """Return whether a cached image can be decoded as a complete image."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def write_png_atomic(image: Image.Image, path: Path) -> None:
    """Encode and publish a rasterized page without exposing partial PNG data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(descriptor)
        image.save(temporary_name, "PNG")
        if not image_file_is_valid(Path(temporary_name)):
            raise RuntimeError(f"rasterizer produced an invalid image for {path.name}")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
