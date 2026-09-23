from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import rasterio

from .model import BANDS

PARTITIONS = {"train", "validation", "test"}
FIELD_ALIASES = {
    "path": ("path", "image_member_path"),
    "imo": ("imo", "vessel_imo"),
    "split": ("split", "partition"),
    "capture_time": ("capture_time", "capture_time_utc"),
}
BAND_ALIASES = {
    "red": "red",
    "b04": "red",
    "green": "green",
    "b03": "green",
    "blue": "blue",
    "b02": "blue",
    "nir": "nir",
    "b08": "nir",
    "coastal": "coastal",
    "b01": "coastal",
    "rededge1": "rededge1",
    "b05": "rededge1",
    "rededge2": "rededge2",
    "b06": "rededge2",
    "rededge3": "rededge3",
    "b07": "rededge3",
    "nir08": "nir08",
    "b8a": "nir08",
    "nir09": "nir09",
    "b09": "nir09",
    "swir16": "swir16",
    "b11": "swir16",
    "swir22": "swir22",
    "b12": "swir22",
}


def _open_text(path: str | Path) -> TextIO:
    path = Path(path)
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    )


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield {**row, "_line": line_number}


def _first(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _geometry_table(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        scene = str(row.get("source_scene_id", "")).strip()
        if not scene or scene in result:
            raise ValueError(f"{path}:{row['_line']}: missing or duplicate source_scene_id")
        result[scene] = {key: value for key, value in row.items() if key != "_line"}
    return result


def read_manifest(
    path: str | Path,
    *,
    crop_root: str | Path | None = None,
    geometry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load compact manifests or the field names used by the Dataverse release."""
    root = Path(crop_root).resolve() if crop_root is not None else None
    scenes = _geometry_table(geometry_path)
    rows = []
    for source in _jsonl(path):
        row = {key: _first(source, aliases) for key, aliases in FIELD_ALIASES.items()}
        row.update({key: value for key, value in source.items() if key not in row})
        if root is not None and row["path"] is not None:
            resolved = (root / str(row["path"])).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{source['_line']}: image path escapes crop root"
                ) from error
            row["path"] = str(resolved)
        scene = str(source.get("source_scene_id", ""))
        if scene and scene in scenes:
            row.update(scenes[scene])
        rows.append(row)
    validate_manifest(rows)
    return rows


def _timestamp(value: Any, line: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"manifest row {line}: invalid capture_time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"manifest row {line}: capture_time must include a UTC offset")
    return parsed


def validate_manifest(rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("manifest is empty")
    partitions: dict[str, str] = {}
    paths: set[str] = set()
    for ordinal, row in enumerate(rows, 1):
        line = row.get("_line", ordinal)
        missing = {name for name in FIELD_ALIASES if row.get(name) in (None, "")}
        if missing:
            raise ValueError(f"manifest row {line}: missing {', '.join(sorted(missing))}")
        split = str(row["split"])
        if split not in PARTITIONS:
            raise ValueError(f"manifest row {line}: invalid split {split!r}")
        _timestamp(row["capture_time"], line)
        path = str(row["path"])
        if path in paths:
            raise ValueError(f"manifest row {line}: duplicate image path {path!r}")
        paths.add(path)
        imo = str(row["imo"])
        previous = partitions.setdefault(imo, split)
        if previous != split:
            raise ValueError(f"identity {imo} crosses partitions")
        geometry(row)


def _band_name(value: Any) -> str:
    normalized = str(value or "").lower().replace("_", "").replace("-", "")
    return BAND_ALIASES.get(normalized, normalized)


def read_crop(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as source:
        names = tuple(_band_name(name) for name in source.descriptions)
        if (source.height, source.width) != (64, 64) or any(
            names.count(name) != 1 for name in BANDS
        ):
            raise ValueError(f"{path}: expected one copy of each named band in a 64 x 64 TIFF")
        values = source.read([names.index(name) + 1 for name in BANDS])
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite pixels")
    return np.clip(values, 0, 10_000).astype(np.float32) / 10_000.0


def _direction(azimuth: float, elevation: float) -> list[float]:
    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)
    cosine = math.cos(elevation_radians)
    return [
        math.sin(azimuth_radians) * cosine,
        -math.cos(azimuth_radians) * cosine,
        math.sin(elevation_radians),
        1.0,
    ]


def _angles(row: Mapping[str, Any], kind: str) -> tuple[Any, Any, bool]:
    if kind == "sun":
        azimuth = row.get("sun_azimuth_deg", row.get("sun_azimuth"))
        elevation = row.get("sun_elevation_deg", row.get("sun_elevation"))
    else:
        azimuth = row.get("view_azimuth_deg", row.get("view_azimuth"))
        elevation = row.get("view_incidence_deg", row.get("view_elevation"))
    valid = bool(row.get(f"{kind}_valid", azimuth is not None and elevation is not None))
    return azimuth, elevation, valid


def geometry(row: Mapping[str, Any]) -> np.ndarray:
    result: list[float] = []
    for kind in ("sun", "view"):
        azimuth, elevation, valid = _angles(row, kind)
        if not valid:
            result.extend([0.0] * 4)
            continue
        if azimuth is None or elevation is None:
            raise ValueError(f"valid {kind} geometry is missing an angle")
        azimuth, elevation = float(azimuth), float(elevation)
        low = -90.0 if kind == "sun" else 0.0
        if not (math.isfinite(azimuth) and 0 <= azimuth <= 360):
            raise ValueError(f"invalid {kind} azimuth")
        if not (math.isfinite(elevation) and low <= elevation <= 90):
            raise ValueError(f"invalid {kind} elevation/incidence")
        result.extend(_direction(azimuth, elevation))
    return np.asarray(result, dtype=np.float32)


def transform_d4(image: np.ndarray, vector: np.ndarray, view: int) -> tuple[np.ndarray, np.ndarray]:
    if image.shape != (12, 64, 64) or vector.shape != (8,) or not 0 <= view < 8:
        raise ValueError("D4 transform expects image [12,64,64], geometry [8] and view 0..7")
    transformed_image = np.rot90(image, view % 4, axes=(-2, -1)).copy()
    transformed_vector = vector.copy()
    for offset in (0, 4):
        x, y = vector[offset : offset + 2]
        rotated = ((x, y), (y, -x), (-x, -y), (-y, x))[view % 4]
        transformed_vector[offset] = -rotated[0] if view >= 4 else rotated[0]
        transformed_vector[offset + 1] = rotated[1]
    if view >= 4:
        transformed_image = transformed_image[..., ::-1].copy()
    return transformed_image, transformed_vector
