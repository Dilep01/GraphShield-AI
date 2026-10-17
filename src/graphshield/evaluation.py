"""Validation-only threshold selection, alert policies and paired time-block intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve


def choose_threshold(y, scores):
    if len(np.unique(y)) != 2:
        raise ValueError("Tuning requires both positive and negative examples")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Prefer the larger threshold when F1 is tied. No test outcomes enter this choice.
    index = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[index])


def classification(y, selected):
    y, selected = np.asarray(y, dtype=bool), np.asarray(selected, dtype=bool)
    tp = int(np.sum(y & selected))
    fp = int(np.sum(~y & selected))
    fn = int(np.sum(y & ~selected))
    tn = int(np.sum(~y & ~selected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def top_k_selection(scores, source_rows, k):
    if k < 1:
        raise ValueError("K must be positive")
    order = np.lexsort((source_rows, -np.asarray(scores)))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:min(k, len(order))]] = True
    return selected


def per_step_selection(scores, steps, source_rows, k):
    if k < 1:
        raise ValueError("Budget must be positive")
    selected = np.zeros(len(scores), dtype=bool)
    for step in np.unique(steps):
        indices = np.flatnonzero(steps == step)
        selected[indices[top_k_selection(np.asarray(scores)[indices], source_rows[indices], k)]] = True
    return selected


def reference_policies(y, steps, budget, training_prevalence):
    """Exact expected yield of uniform random selection within each time bucket."""
    if budget < 1:
        raise ValueError("Budget must be positive")
    y, steps = np.asarray(y), np.asarray(steps)
    expected_tp, alerts, small_steps, positives_in_small_steps = 0.0, 0, 0, 0
    for step in np.unique(steps):
        labels = y[steps == step]
        count = min(budget, len(labels))
        alerts += count
        expected_tp += count * float(labels.mean())
        if len(labels) <= budget:
            small_steps += 1
            positives_in_small_steps += int(labels.sum())
    return {
        "constant_score_average_precision": float(y.mean()),
        "constant_training_prevalence_brier": float(np.mean((y - training_prevalence) ** 2)),
        "training_prevalence": training_prevalence,
        "random_within_step_budget": {
            "expected_true_positives": expected_tp, "alerts": alerts,
            "expected_precision": expected_tp / alerts if alerts else None,
            "expected_recall": expected_tp / y.sum() if y.sum() else None,
            "steps_with_rows_at_or_below_budget": small_steps,
            "positives_in_those_steps": positives_in_small_steps,
            "meaning": "Analytical random-selection expectation, not a fitted model or observed result",
        },
    }


def evaluate(y, scores, steps, source_rows, threshold, budget, top_k):
    y, scores = np.asarray(y), np.asarray(scores)
    if not len(y):
        return {"rows": 0, "status": "empty subgroup"}
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite values in [0,1]")
    result = {"rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
              "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
              "brier_score": float(brier_score_loss(y, scores)),
              "log_loss": float(log_loss(y, scores, labels=[0, 1])),
              "threshold": threshold, "threshold_policy": classification(y, scores >= threshold),
              "global_top_k": {"k": top_k, **classification(y, top_k_selection(scores, source_rows, top_k))},
              "per_step_budget": {"k": budget, **classification(y, per_step_selection(scores, steps, source_rows, budget))}}
    bins = np.minimum((scores * 10).astype(int), 9)
    result["reliability_bins"] = [{
        "lower": index / 10, "upper": (index + 1) / 10,
        "rows": int((bins == index).sum()),
        "mean_score": float(scores[bins == index].mean()),
        "observed_rate": float(y[bins == index].mean()),
    } for index in range(10) if np.any(bins == index)]
    return result


def paired_block_intervals(y, score_map, steps, repeats, seed):
    """Sample complete observed steps with replacement, paired across every model."""
    unique_steps, inverse = np.unique(steps, return_inverse=True)
    if repeats < 1 or len(unique_steps) < 2:
        return {"status": "insufficient resamples or time blocks"}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in score_map}
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(unique_steps), len(unique_steps)), minlength=len(unique_steps))
        weights = counts[inverse]
        if np.dot(weights, y) == 0:
            continue
        for name, scores in score_map.items():
            samples[name].append(float(average_precision_score(y, scores, sample_weight=weights)))
    if not all(samples.values()):
        return {"status": "no resamples with positive outcomes"}
    result = {"method": "Paired bootstrap of whole observed time steps; percentile 95% intervals",
              "limitation": "Assumes step blocks are exchangeable; does not model dependence across adjacent steps",
              "repeats_requested": repeats, "valid_repeats": len(next(iter(samples.values()))),
              "seed": seed, "average_precision": {
                  name: [float(x) for x in np.quantile(values, [.025, .975])]
                  for name, values in samples.items()}}
    names = list(samples)
    if len(names) == 2:
        delta = np.asarray(samples[names[1]]) - np.asarray(samples[names[0]])
        result["difference"] = {"comparison": f"{names[1]} minus {names[0]}",
                                "interval": [float(x) for x in np.quantile(delta, [.025, .975])]}
    return result
