# Sentinel-2 vessel re-identification

Reproducibility code for *Fleet-scale identification of individual large ships
from Sentinel-2 observations*. The package implements the reported encoder,
training objective, eight-view inference and chronological retrieval evaluation.

The accompanying Harvard Dataverse deposit provides the twelve-band crops,
identity partitions, scene-angle metadata and Source Data. The GitHub release
asset `sentinel2-vessel-reid-models-v1.0.0.zip` contains the selected model and
its RGB-pilot training initialization. Verify the archive with
`weights/RELEASE_SHA256SUMS`, then verify both extracted checkpoints with
`weights/SHA256SUMS`.

## Installation

Python 3.10–3.12 is supported. PyTorch 2.7.1 reproduces the reported run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Inputs

The commands accept either the Dataverse observation fields or the shorter
aliases shown below.

| Dataverse field | Alias |
| --- | --- |
| `image_member_path` | `path` |
| `vessel_imo` | `imo` |
| `partition` | `split` |
| `capture_time_utc` | `capture_time` |

Pass the directory containing extracted TIFF members with `--crop-root`. Pass
the released scene-geometry JSONL file with `--geometry`. TIFFs must be 64 × 64
and contain the twelve named Sentinel-2 bands. Digital numbers are clipped to
0–10,000, divided by 10,000 and log-scaled inside the model. They are not
converted to surface reflectance.

```bash
vessel-reid validate-manifest metadata/observations/observations-000000.jsonl.gz \
  --crop-root /path/to/extracted-crops \
  --geometry metadata/scene-geometry/scene-geometry.jsonl.gz
```

Several observation shards can be concatenated into one JSONL file before use.
Validation checks timestamps, identity-disjoint partitions, duplicate image
paths and scene-geometry values.

## Inference and evaluation

```bash
vessel-reid infer manifest.jsonl.gz weights/model.pt features/test.npz \
  --split test --device mps --crop-root /path/to/crops \
  --geometry scene-geometry.jsonl.gz

vessel-reid evaluate features/test.npz \
  --cutoff 2025-05-29T00:00:00Z --gap-days 7 --depth 20
```

The cutoff separates earlier gallery observations from later queries. `--depth`
sets the maximum earlier references per identity, with `0` selecting the full
gallery. Scoring uses the mean squared distance to the two nearest references
and reports query-level and identity-macro retrieval metrics with the reported
2,000-replicate identity-bootstrap intervals. Use `infer --split all` for a
gallery spanning every partition.

## Training

```bash
vessel-reid train manifest.jsonl.gz weights/rgb-pilot.pt runs/train \
  --device mps --epochs 10 --crop-root /path/to/crops \
  --geometry scene-geometry.jsonl.gz
```

The frozen recipe uses identity-disjoint data, coverage epochs, P=16 and K=4,
global identity proxies, proxy cross-entropy plus supervised contrastive loss,
D4 augmentation, AdamW and the reported learning-rate schedule. Select a
checkpoint using validation data only. Exact reproduction requires the released
data, initialization and hashes, as well as the locked environment in
`requirements-lock.txt`.

## Scope and licence

This compact repository covers the paper's central model and retrieval path.
Source Data in Dataverse contain the outputs of the additional controls and
post-hoc analyses. Code is MIT licensed. Separately distributed model weights
are CC BY 4.0; dataset terms are stated in the Dataverse deposit.
