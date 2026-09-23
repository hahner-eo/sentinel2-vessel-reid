from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import geometry, read_crop, transform_d4
from .model import load_model


def objective(embedding, proxy, labels, sample_keys, weights):
    """Scale-10 proxy CE plus 0.1-weighted supervised contrastive loss."""
    normalized = F.normalize(embedding, dim=1)
    logits = 10 * normalized @ F.normalize(proxy, dim=1).T
    proxy_ce = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()
    independent = sample_keys[:, None].ne(sample_keys[None, :]) & weights[None, :].gt(0)
    positive = labels[:, None].eq(labels[None, :]) & independent
    contrastive_logits = (normalized @ normalized.T / 0.1).masked_fill(~independent, -1e4)
    log_probability = contrastive_logits - torch.logsumexp(contrastive_logits, 1, keepdim=True)
    positive_count = positive.sum(1)
    supervised_contrastive = (
        -(log_probability * positive).sum(1) / positive_count.clamp_min(1) * weights
    ).mean()
    return proxy_ce + 0.1 * supervised_contrastive


def coverage_batches(groups, rng, *, p=16, k=4):
    """Visit every observation once; use zero-weight padding for fixed batches."""
    if len(groups) < p or any(len(group) == 0 for group in groups):
        raise ValueError(f"coverage sampling requires at least {p} non-empty identities")
    orders = [rng.permutation(group) for group in groups]
    chunks = [
        (identity, start)
        for identity, group in enumerate(groups)
        for start in range(0, len(group), k)
    ]
    chunks = [chunks[index] for index in rng.permutation(len(chunks))]
    mean_count = float(np.mean([len(group) for group in groups]))
    for start in range(0, len(chunks), p):
        real = chunks[start : start + p]
        padded = real + [real[0]] * (p - len(real))
        rows, labels, weights = [], [], []
        for ordinal, (identity, offset) in enumerate(padded):
            selected = orders[identity][offset : offset + k].tolist()
            genuine = len(selected) if ordinal < len(real) else 0
            selected += [selected[0]] * (k - len(selected))
            rows.extend(selected)
            labels.extend([identity] * k)
            weights.extend([mean_count / len(groups[identity])] * genuine + [0.0] * (k - genuine))
        yield (
            np.asarray(rows),
            np.asarray(labels),
            np.asarray(weights, dtype=np.float32),
        )


def _parameter_groups(model, proxy):
    """Apply the reported learning rates and weight-decay exclusions."""
    rates = {"backbone": 1e-5, "spectral": 1e-4, "geometry": 1e-4, "head": 3e-4}
    groups = {}
    seen = set()
    for name, parameter in [*model.named_parameters(), ("proxy", proxy)]:
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        if name == "spectral_patch.weight":
            family = "spectral"
        elif name.startswith("geometry_film."):
            family = "geometry"
        elif name.startswith("head.") or name == "proxy":
            family = "head"
        else:
            family = "backbone"
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or name == "proxy"
            or any(token in name for token in ("pos_embed", "cls_token", "mask_token"))
        )
        key = family, no_decay
        group = groups.setdefault(
            key,
            {
                "params": [],
                "lr": rates[family],
                "weight_decay": 0.0 if no_decay else 0.05,
                "name": f"{family}_{'nodecay' if no_decay else 'decay'}",
            },
        )
        group["params"].append(parameter)
    return list(groups.values())


def _identity_groups(rows):
    identities = sorted({str(row["imo"]) for row in rows})
    index = {imo: ordinal for ordinal, imo in enumerate(identities)}
    groups = [[] for _ in identities]
    for row_index, row in enumerate(rows):
        groups[index[str(row["imo"])]].append(row_index)
    return identities, groups


def _device_rng_state(device):
    device = torch.device(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return None


def _set_device_rng_state(device, state):
    if state is None:
        return
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.set_rng_state(state)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state, device)


def _initialize_proxy(identity_count, device, seed):
    host_state = torch.get_rng_state()
    device_state = _device_rng_state(device)
    try:
        torch.manual_seed(seed)
        return torch.nn.Parameter(torch.randn(identity_count, 256, device=device) / math.sqrt(256))
    finally:
        torch.set_rng_state(host_state)
        _set_device_rng_state(device, device_state)


def train(
    rows,
    initialization,
    output,
    *,
    device="cpu",
    epochs=10,
    seed=20260911,
    resume=None,
):
    train_rows = [row for row in rows if row["split"] == "train"]
    identities, groups = _identity_groups(train_rows)
    if len(groups) < 16:
        raise ValueError("training needs at least 16 identities")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    if torch.device(device).type == "mps":
        torch.mps.manual_seed(seed)
    elif torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed)
    model = load_model(initialization, device, allow_missing_adapters=True).train()
    proxy = _initialize_proxy(len(identities), device, seed + 7)
    optimizer = torch.optim.AdamW(_parameter_groups(model, proxy))
    rng = np.random.default_rng(seed)
    completed_epoch = completed_steps = 0
    torch.manual_seed(seed)

    steps_per_epoch = math.ceil(sum(math.ceil(len(group) / 4) for group in groups) / 16)
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(2000, max(total_steps - 1, 0))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("identities") != identities or saved.get("seed") != seed:
            raise ValueError("resume checkpoint does not match the manifest or seed")
        model.load_state_dict(saved["model"])
        proxy.data.copy_(saved["proxy"].to(device))
        optimizer.load_state_dict(saved["optimizer"])
        completed_epoch, completed_steps = saved["epoch"], saved["step"]
        rng.bit_generator.state = saved["numpy_rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        _set_device_rng_state(device, saved.get("device_rng_state"))
        if completed_epoch >= epochs:
            raise ValueError("resume checkpoint already reached the requested epoch")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(completed_epoch + 1, epochs + 1):
        loss = None
        for indices, class_indices, anchor_weights in coverage_batches(groups, rng):
            update = completed_steps + 1
            if warmup_steps and update <= warmup_steps:
                factor = update / warmup_steps
            else:
                progress = (update - warmup_steps) / max(total_steps - warmup_steps, 1)
                factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr * factor

            transformed = [
                transform_d4(
                    read_crop(train_rows[index]["path"]),
                    geometry(train_rows[index]),
                    int(rng.integers(8)),
                )
                for index in indices
            ]
            images, angles = zip(*transformed, strict=True)
            labels = torch.as_tensor(class_indices, dtype=torch.long, device=device)
            keys = torch.as_tensor(indices, dtype=torch.long, device=device)
            weights = torch.as_tensor(anchor_weights, dtype=torch.float32, device=device)
            embedded = model(
                torch.from_numpy(np.stack(images)).to(device),
                torch.from_numpy(np.stack(angles)).to(device),
            )
            loss = objective(embedded, proxy, labels, keys, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed_steps += 1

        checkpoint = {
            "model": model.state_dict(),
            "proxy": proxy.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": completed_steps,
            "identities": identities,
            "seed": seed,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "device_rng_state": _device_rng_state(device),
        }
        torch.save(checkpoint, output / f"epoch{epoch}.pt")
        summary = {"epoch": epoch, "step": completed_steps, "loss": float(loss.detach())}
        (output / "latest.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
