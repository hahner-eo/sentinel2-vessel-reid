from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import geometry, read_crop, transform_d4
from .model import load_model


def _views(rows):
    images, angles = [], []
    for row in rows:
        image, vector = read_crop(row["path"]), geometry(row)
        for view in range(8):
            pixels, transformed = transform_d4(image, vector, view)
            images.append(pixels)
            angles.append(transformed)
    return np.stack(images), np.stack(angles)


def embed_manifest(rows, checkpoint, output, *, split, device="cpu", batch_size=64):
    """Embed one partition using an unaugmented gallery and eight-view queries."""
    if batch_size < 8:
        raise ValueError("batch_size must be at least eight")
    selected = list(rows) if split == "all" else [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no rows for split {split}")
    model = load_model(checkpoint, device)
    gallery_embeddings, query_embeddings = [], []
    rows_per_batch = max(1, batch_size // 8)
    with torch.inference_mode():
        for start in range(0, len(selected), rows_per_batch):
            batch = selected[start : start + rows_per_batch]
            images, angles = _views(batch)
            embedded = model(
                torch.from_numpy(images).to(device),
                torch.from_numpy(angles).to(device),
            ).reshape(len(batch), 8, -1)
            gallery_embeddings.append(embedded[:, 0].cpu().numpy())
            query_embeddings.append(F.normalize(embedded.mean(1), dim=-1).cpu().numpy())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gallery_embeddings=np.concatenate(gallery_embeddings).astype(np.float32),
        query_embeddings=np.concatenate(query_embeddings).astype(np.float32),
        imo=np.asarray([str(row["imo"]) for row in selected]),
        capture_time=np.asarray([row["capture_time"] for row in selected]),
        path=np.asarray([row["path"] for row in selected]),
    )
