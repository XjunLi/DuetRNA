from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .features import RelationRecord


EPS = 1e-6
CANONICAL_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("G", "C"),
    ("C", "G"),
    ("G", "U"),
    ("U", "G"),
}


@dataclass(frozen=True)
class GaussianStats:
    mean: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    logdet: float
    n: int
    q95_mahalanobis: float


def _record_label(rec: RelationRecord, label_key: str) -> str:
    if label_key == "relation":
        return rec.relation
    if label_key == "family":
        return rec.family
    raise ValueError(f"unknown label_key: {label_key}")


def _feature_matrix(
    records: list[RelationRecord],
    frame: str,
    family: str | None = None,
    *,
    label_key: str = "family",
) -> tuple[np.ndarray, list[str]]:
    xs = []
    labels = []
    for rec in records:
        if family is not None and rec.family != family:
            continue
        value = getattr(rec, f"z_{frame}")
        if value is None:
            continue
        xs.append(value)
        labels.append(_record_label(rec, label_key))
    if not xs:
        return np.zeros((0, 6), dtype=np.float64), []
    return np.stack(xs, axis=0), labels


def _regularized_cov(x: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64)
    cov = np.cov(x, rowvar=False)
    return cov + np.eye(cov.shape[0], dtype=np.float64) * ridge


def fit_gaussian_stats(x: np.ndarray) -> GaussianStats:
    mean = x.mean(axis=0)
    cov = _regularized_cov(x)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(cov.shape[0]) * 1e-3
        sign, logdet = np.linalg.slogdet(cov)
    inv_cov = np.linalg.pinv(cov)
    delta = x - mean
    d2 = np.einsum("nd,dd,nd->n", delta, inv_cov, delta)
    q95 = float(np.quantile(d2, 0.95)) if len(d2) else float("nan")
    return GaussianStats(mean=mean, cov=cov, inv_cov=inv_cov, logdet=float(logdet), n=x.shape[0], q95_mahalanobis=q95)


def fit_relation_gaussians(records: list[RelationRecord], frame: str, *, label_key: str = "relation") -> dict[str, GaussianStats]:
    by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        value = getattr(rec, f"z_{frame}")
        if value is not None:
            by_family[_record_label(rec, label_key)].append(value)
    out = {}
    for family, values in by_family.items():
        if len(values) >= 2:
            out[family] = fit_gaussian_stats(np.stack(values, axis=0))
    return out


def relation_compactness(records: list[RelationRecord], frame: str, *, label_key: str = "relation") -> dict[str, float]:
    """Relation Compactness Score (RCS): weighted logdet covariance.

    Lower is better. Per-family values are also returned for diagnostics.
    """
    stats = fit_relation_gaussians(records, frame, label_key=label_key)
    total = sum(item.n for item in stats.values())
    out: dict[str, float] = {}
    weighted = 0.0
    for family, item in stats.items():
        weight = item.n / max(total, 1)
        out[f"rcs_{frame}_{family}"] = item.logdet
        weighted += weight * item.logdet
    out[f"rcs_{frame}"] = weighted if total else float("nan")
    out[f"rcs_{frame}_n"] = float(total)
    return out


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return float("nan")
    f1s = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def relation_separability(records: list[RelationRecord], frame: str, *, seed: int = 123, label_key: str = "relation") -> dict[str, float]:
    """Nearest-centroid Relation Separability Score (RSS).

    This intentionally avoids a heavy classifier dependency. The metric uses
    only relative-pose vectors and reports held-out macro-F1/accuracy.
    """
    x, labels = _feature_matrix(records, frame, label_key=label_key)
    if x.shape[0] < 10 or len(set(labels)) < 2:
        return {f"rss_{frame}_macro_f1": float("nan"), f"rss_{frame}_accuracy": float("nan"), f"rss_{frame}_n": float(x.shape[0])}
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = max(1, int(round(0.8 * len(idx))))
    train_idx, test_idx = idx[:split], idx[split:]
    if len(test_idx) == 0:
        test_idx = train_idx
    centroids = {}
    for label in sorted(set(labels)):
        label_idx = [i for i in train_idx if labels[i] == label]
        if label_idx:
            centroids[label] = x[label_idx].mean(axis=0)
    y_pred = []
    y_true = [labels[i] for i in test_idx]
    for i in test_idx:
        pred = min(centroids, key=lambda label: float(np.sum((x[i] - centroids[label]) ** 2)))
        y_pred.append(pred)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {
        f"rss_{frame}_macro_f1": _macro_f1(y_true, y_pred),
        f"rss_{frame}_accuracy": float(acc),
        f"rss_{frame}_n": float(x.shape[0]),
    }


def _mahalanobis_d2(x: np.ndarray, stats: GaussianStats) -> np.ndarray:
    delta = x - stats.mean
    return np.einsum("nd,dd,nd->n", delta, stats.inv_cov, delta)


def bpgv_metrics(records: list[RelationRecord], reference: dict[str, GaussianStats], frame: str = "base") -> dict[str, float]:
    """Base Pair Geometry Validity against reference relative-pose Gaussians."""
    by_relation: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        if rec.family != "pair" and "pair" not in rec.family:
            continue
        value = getattr(rec, f"z_{frame}")
        if value is not None:
            by_relation[rec.relation].append(value)
    out: dict[str, float] = {}
    total_valid = 0
    total_count = 0
    nll_terms = []
    for relation, values in by_relation.items():
        ref = reference.get(relation) or reference.get("pair")
        if ref is None or not values:
            continue
        x = np.stack(values, axis=0)
        d2 = _mahalanobis_d2(x, ref)
        valid = d2 <= ref.q95_mahalanobis
        out[f"bpgv_{relation}"] = float(valid.mean())
        out[f"bpgv_{relation}_n"] = float(len(values))
        total_valid += int(valid.sum())
        total_count += len(values)
        nll_terms.extend((0.5 * (d2 + ref.logdet + x.shape[1] * math.log(2 * math.pi))).tolist())
    out["bpgv"] = total_valid / max(total_count, 1) if total_count else float("nan")
    out["pair_geometry_nll"] = float(np.mean(nll_terms)) if nll_terms else float("nan")
    out["bpgv_n"] = float(total_count)
    return out


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def stack_reference(records: list[RelationRecord]) -> dict[str, np.ndarray]:
    values = [rec.stack_features for rec in records if rec.family == "stack" and rec.stack_features is not None]
    if not values:
        return {}
    x = np.stack(values, axis=0)
    return {
        "mean": x.mean(axis=0),
        "std": x.std(axis=0) + EPS,
        "q05": np.quantile(x, 0.05, axis=0),
        "q95": np.quantile(x, 0.95, axis=0),
        "values": x,
    }


def stack_geometry_metrics(records: list[RelationRecord], reference: dict[str, np.ndarray]) -> dict[str, float]:
    values = [rec.stack_features for rec in records if rec.family == "stack" and rec.stack_features is not None]
    if not values or "q05" not in reference:
        return {
            "bsgv": float("nan"),
            "base_stacking_geometry_validity": float("nan"),
            "stack_valid": float("nan"),
            "stack_validity": float("nan"),
            "stack_emd": float("nan"),
            "stack_n": 0.0,
        }
    x = np.stack(values, axis=0)
    lo, hi = reference["q05"], reference["q95"]
    valid = np.all((x >= lo) & (x <= hi), axis=1)
    ref_values = reference.get("values")
    emds = []
    if ref_values is not None:
        for dim in range(x.shape[1]):
            emds.append(_wasserstein_1d(x[:, dim], ref_values[:, dim]))
    z = (x - reference["mean"]) / reference["std"]
    stack_valid = float(valid.mean())
    stack_emd = float(np.nanmean(emds)) if emds else float("nan")
    return {
        # Naming aliases are intentional: slide-facing reports use BSGV, while
        # Earlier metric exports used StackValid/stack_valid.
        "bsgv": stack_valid,
        "base_stacking_geometry_validity": stack_valid,
        "stack_valid": stack_valid,
        "stack_validity": stack_valid,
        "stack_emd": stack_emd,
        "stack_z_mse": float(np.mean(z**2)),
        "stack_n": float(len(values)),
    }


def relation_self_consistency(gen_records: list[RelationRecord], folded_records: list[RelationRecord]) -> dict[str, float]:
    def normalize_pdb_name(name: str) -> str:
        """Map forward-folded names back to generated sample names.

        EvalSuite writes direct RhoFold predictions as
        ``50_na_sample_15_0.pdb`` while the generated sample is
        ``50_na_sample_15.pdb``.  Without this normalization, RSC would compare
        disjoint key sets and incorrectly report zero overlap.
        """
        stem = str(name)
        if stem.endswith(".pdb"):
            stem = stem[:-4]
        match = re.match(r"^(.+_na_sample_\d+)_(?:seq)?\d+$", stem)
        if match:
            return match.group(1)
        match = re.match(r"^(.+)__seq\d+(?:_model_\d+)?$", stem)
        if match:
            return match.group(1)
        return stem

    def keys(records: list[RelationRecord], family: str | None = None) -> set[tuple[str, int, int, str]]:
        out = set()
        for rec in records:
            if family is not None and rec.family != family:
                continue
            a, b = sorted((rec.i, rec.j))
            out.add((normalize_pdb_name(rec.pdb_name), a, b, rec.family))
        return out

    metrics = {}
    for family in ("pair", "stack", None):
        a = keys(gen_records, family)
        b = keys(folded_records, family)
        tp = len(a & b)
        precision = tp / max(len(a), 1)
        recall = tp / max(len(b), 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        prefix = "rsc_rel" if family is None else f"rsc_{family}"
        metrics[f"{prefix}_precision"] = float(precision)
        metrics[f"{prefix}_recall"] = float(recall)
        metrics[f"{prefix}_f1"] = float(f1)
        metrics[f"{prefix}_gen_n"] = float(len(a))
        metrics[f"{prefix}_folded_n"] = float(len(b))
    return metrics


def sequence_base_compatibility(records: list[RelationRecord], data_counts: dict[str, Counter] | None = None) -> dict[str, float]:
    pair_records = [rec for rec in records if rec.family == "pair" or "pair" in rec.family]
    if not pair_records:
        return {"sbc_canonical_compat": float("nan"), "sbc_log_likelihood": float("nan"), "sbc_n": 0.0}
    canonical_hits = 0
    canonical_total = 0
    logps = []
    for rec in pair_records:
        pair = (rec.base_i, rec.base_j)
        if rec.relation.lower() in {"cww", "wc", "canonical", "pair_candidate", "pair"} or rec.relation.endswith("WW"):
            canonical_total += 1
            canonical_hits += int(pair in CANONICAL_PAIRS)
        if data_counts is not None:
            counts = data_counts.get(rec.relation) or data_counts.get(rec.family)
            if counts:
                total = sum(counts.values()) + 16
                logps.append(math.log((counts.get("".join(pair), 0) + 1) / total))
    return {
        "sbc_canonical_compat": canonical_hits / max(canonical_total, 1) if canonical_total else float("nan"),
        "sbc_log_likelihood": float(np.mean(logps)) if logps else float("nan"),
        "sbc_n": float(len(pair_records)),
    }


def base_pair_counts(records: list[RelationRecord]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        if rec.family == "pair" or "pair" in rec.family:
            out[rec.family]["".join((rec.base_i, rec.base_j))] += 1
            out[rec.relation]["".join((rec.base_i, rec.base_j))] += 1
    return dict(out)
