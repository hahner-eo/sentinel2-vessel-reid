import gzip
import json
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

from vessel_reid.data import (
    BANDS,
    geometry,
    read_crop,
    read_manifest,
    transform_d4,
    validate_manifest,
)
from vessel_reid.evaluate import evaluate_archive
from vessel_reid.model import GeometryFiLM, VesselEncoder
from vessel_reid.training import _parameter_groups, coverage_batches


def test_model_shape():
    model = VesselEncoder(65).eval()
    with torch.inference_mode():
        value = model(torch.zeros(2, 12, 64, 64), torch.zeros(2, 8))
    assert value.shape == (2, 256)
    assert torch.allclose(value.norm(dim=1), torch.ones(2), atol=1e-5)


def test_whole_record_geometry_dropout(monkeypatch):
    module = GeometryFiLM().train()
    for parameter in module.parameters():
        torch.nn.init.constant_(parameter, 0.2)
    monkeypatch.setattr(torch, "rand", lambda shape, device=None: torch.zeros(shape, device=device))
    tokens = torch.randn(2, 64, 384)
    vectors = torch.tensor([[1, 0, 0, 1, 0, 1, 0, 1]] * 2, dtype=torch.float32)
    assert torch.equal(module(tokens, vectors), tokens)


def test_d4_identity_and_partition_guard():
    image = np.arange(12 * 64 * 64, dtype=np.float32).reshape(12, 64, 64)
    vector = np.asarray([1, 0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    actual, transformed = transform_d4(image, vector, 0)
    assert np.array_equal(actual, image)
    assert np.array_equal(transformed, vector)
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        {"path": "a", "imo": "1", "split": "train", "capture_time": now},
        {"path": "b", "imo": "1", "split": "test", "capture_time": now},
    ]
    with pytest.raises(ValueError, match="crosses partitions"):
        validate_manifest(rows)


def test_dataverse_manifest_and_geometry(tmp_path):
    manifest = tmp_path / "observations.jsonl.gz"
    scene_geometry = tmp_path / "geometry.jsonl.gz"
    observation = {
        "image_member_path": "objects/example.tif",
        "vessel_imo": "9313149",
        "partition": "test",
        "capture_time_utc": "2025-01-01T00:00:00Z",
        "source_scene_id": "scene-1",
    }
    angles = {
        "source_scene_id": "scene-1",
        "sun_valid": True,
        "view_valid": False,
        "sun_azimuth_deg": 90,
        "sun_elevation_deg": 0,
        "view_azimuth_deg": None,
        "view_incidence_deg": None,
    }
    with gzip.open(manifest, "wt") as stream:
        stream.write(json.dumps(observation) + "\n")
    with gzip.open(scene_geometry, "wt") as stream:
        stream.write(json.dumps(angles) + "\n")
    rows = read_manifest(manifest, crop_root=tmp_path, geometry_path=scene_geometry)
    assert rows[0]["imo"] == "9313149"
    assert rows[0]["path"] == str((tmp_path / "objects/example.tif").resolve())
    assert np.allclose(geometry(rows[0]), [1, 0, 0, 1, 0, 0, 0, 0], atol=1e-6)


def test_read_crop_accepts_sentinel_band_names(tmp_path):
    path = tmp_path / "crop.tif"
    sentinel_names = (
        "B04",
        "B03",
        "B02",
        "B08",
        "B01",
        "B05",
        "B06",
        "B07",
        "B8A",
        "B09",
        "B11",
        "B12",
    )
    values = np.arange(12, dtype=np.uint16)[:, None, None] * np.ones((12, 64, 64), np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=12,
        dtype="uint16",
        transform=Affine(10, 0, 500_000, 0, -10, 1_000_000),
    ) as target:
        target.write(values)
        target.descriptions = sentinel_names
    actual = read_crop(path)
    assert actual.shape == (len(BANDS), 64, 64)
    assert np.allclose(actual[:, 0, 0], np.arange(12) / 10_000)


def test_evaluation(tmp_path):
    archive = tmp_path / "features.npz"
    features = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32)
    np.savez(
        archive,
        gallery_embeddings=features,
        query_embeddings=features,
        imo=np.asarray(["a", "b", "a", "b"]),
        capture_time=np.asarray(
            [
                "2020-01-01T00:00:00+00:00",
                "2020-01-01T00:00:00+00:00",
                "2021-01-01T00:00:00+00:00",
                "2021-01-01T00:00:00+00:00",
            ]
        ),
    )
    result = evaluate_archive(archive, "2020-06-01T00:00:00+00:00", gap_days=0, depth=20)
    assert result["rank1"] == 100.0
    assert result["macro_rank1"] == 100.0
    assert result["enrolled_candidates"] == 2


def test_coverage_sampler_visits_every_row_once():
    groups = [list(range(index * 5, index * 5 + 5)) for index in range(17)]
    seen = []
    for rows, _, weights in coverage_batches(groups, np.random.default_rng(7)):
        seen.extend(rows[weights > 0].tolist())
        assert len(rows) == 64
    assert sorted(seen) == list(range(85))


def test_optimizer_excludes_bias_norm_tokens_and_proxy_from_decay():
    model = VesselEncoder(65)
    proxy = torch.nn.Parameter(torch.randn(20, 256))
    groups = _parameter_groups(model, proxy)
    decay = {
        id(parameter): group["weight_decay"] for group in groups for parameter in group["params"]
    }
    named = dict(model.named_parameters())
    assert decay[id(named["blocks.0.attn.qkv.weight"])] == 0.05
    assert decay[id(named["blocks.0.attn.qkv.bias"])] == 0.0
    assert decay[id(named["blocks.0.norm1.weight"])] == 0.0
    assert decay[id(named["pos_embed"])] == 0.0
    assert decay[id(proxy)] == 0.0
