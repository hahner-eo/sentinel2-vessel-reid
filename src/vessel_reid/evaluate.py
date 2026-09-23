from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps and cutoff must include a UTC offset")
    return parsed


def _score(query: np.ndarray, gallery: np.ndarray) -> float:
    if not len(gallery):
        return -np.inf
    distances = np.square(gallery - query).sum(1) / query.shape[0]
    nearest = min(2, len(distances))
    return -float(np.partition(distances, nearest - 1)[:nearest].mean())


def _macro_intervals(ranks, groups, *, replicates, seed):
    if replicates < 1:
        return {}
    identities = list(groups)
    result = {}
    for rank_limit in (1, 5):
        generator = np.random.default_rng(seed)
        rates = np.asarray(
            [np.mean(ranks[groups[imo]] <= rank_limit) for imo in identities], dtype=np.float64
        )
        draws = []
        for start in range(0, replicates, 100):
            size = min(100, replicates - start)
            selected = generator.integers(0, len(rates), size=(size, len(rates)))
            draws.extend(rates[selected].mean(1))
        result[f"macro_rank{rank_limit}_ci95"] = (100 * np.quantile(draws, (0.025, 0.975))).tolist()
    return result


def evaluate_archive(
    path,
    cutoff,
    *,
    gap_days=7,
    depth=20,
    bootstrap_replicates=2000,
    seed=20260919,
):
    """Evaluate the paper's past-only chronological retrieval protocol."""
    if gap_days < 0 or depth < 0 or bootstrap_replicates < 0:
        raise ValueError("gap_days, depth and bootstrap_replicates must be non-negative")
    with np.load(path, allow_pickle=False) as data:
        gallery_features = data["gallery_embeddings"]
        query_features = data["query_embeddings"]
        imos = data["imo"].astype(str)
        times = np.asarray([_time(value) for value in data["capture_time"].astype(str)])
    if (
        gallery_features.shape != query_features.shape
        or gallery_features.ndim != 2
        or not (len(imos) == len(times) == len(gallery_features))
    ):
        raise ValueError("feature archive arrays are not aligned")
    if not np.isfinite(gallery_features).all() or not np.isfinite(query_features).all():
        raise ValueError("feature archive contains non-finite values")

    boundary = _time(cutoff)
    identities = sorted(set(imos))
    identity_index = {imo: index for index, imo in enumerate(identities)}
    galleries = []
    enrolled = set()
    for imo in identities:
        indices = np.flatnonzero((imos == imo) & (times < boundary))
        indices = indices[np.argsort(times[indices])]
        if depth:
            indices = indices[-depth:]
        galleries.append(gallery_features[indices])
        if len(indices):
            enrolled.add(imo)

    query_indices = np.flatnonzero(times >= boundary + timedelta(days=gap_days))
    ranks, query_imos = [], []
    for index in query_indices:
        scores = np.asarray([_score(query_features[index], gallery) for gallery in galleries])
        imo = imos[index]
        if imo not in enrolled:
            rank = len(identities) + 1
        else:
            truth_score = scores[identity_index[imo]]
            rank = 1 + int(np.sum(scores > truth_score))
        ranks.append(rank)
        query_imos.append(imo)
    if not ranks:
        raise ValueError("no queries occur after the cutoff and gap")

    ranks_array = np.asarray(ranks)
    query_imos_array = np.asarray(query_imos)
    grouped = {
        imo: np.flatnonzero(query_imos_array == imo) for imo in sorted(set(query_imos_array))
    }
    result = {
        "candidates": len(identities),
        "enrolled_candidates": len(enrolled),
        "queries": len(ranks_array),
        "query_identities": len(grouped),
        "unenrolled_query_identities": len(set(query_imos) - enrolled),
        "mrr": float(np.mean(np.where(ranks_array <= len(identities), 1 / ranks_array, 0))),
        "macro_mrr": float(
            np.mean(
                [
                    np.mean(
                        np.where(
                            ranks_array[indices] <= len(identities),
                            1 / ranks_array[indices],
                            0,
                        )
                    )
                    for indices in grouped.values()
                ]
            )
        ),
    }
    for rank_limit in (1, 5, 10, 100):
        result[f"rank{rank_limit}"] = float(np.mean(ranks_array <= rank_limit) * 100)
        result[f"macro_rank{rank_limit}"] = float(
            np.mean([np.mean(ranks_array[indices] <= rank_limit) for indices in grouped.values()])
            * 100
        )
    result.update(
        _macro_intervals(
            ranks_array,
            grouped,
            replicates=bootstrap_replicates,
            seed=seed,
        )
    )
    result["bootstrap"] = {
        "replicates": bootstrap_replicates,
        "seed": seed,
        "unit": "IMO identity",
    }
    return result
