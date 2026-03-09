def norm_verdict(x):
    x = (x or "").strip().lower()
    if x in {"supported", "unsupported"}:
        return x
    elif x in {"inconclusive", "contradicted", "contradictory", "incorrect", "refuted", "false"}:
        return "unsupported"
    return ""  # blank/invalid

def calculate_scores(human_verdicts, model_verdicts):
    if len(human_verdicts) != len(model_verdicts):
        raise ValueError("human_verdicts and model_verdicts must have the same length")

    y_true = [norm_verdict(t) for t in human_verdicts]        # only "supported"/"unsupported"
    y_pred = [norm_verdict(p) for p in model_verdicts]        # may include ""

    # Requested gold counts
    supported_count = sum(t == "supported"   for t in y_true)
    unsupported_cnt = sum(t == "unsupported" for t in y_true)

    # Confusion matrix (positive class = "supported")
    tp = sum((t == "supported")   and (p == "supported")    for t, p in zip(y_true, y_pred))
    fn = sum((t == "supported")   and (p != "supported")    for t, p in zip(y_true, y_pred))  # blank → FN
    fp = sum((t == "unsupported") and (p != "unsupported")  for t, p in zip(y_true, y_pred))  # blank → FP
    tn = sum((t == "unsupported") and (p == "unsupported")  for t, p in zip(y_true, y_pred))

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n         = len(y_true)
    accuracy  = (tp + tn) / n if n else 0.0

    return {
        "supported_count": supported_count,
        "unsupported_cnt": unsupported_cnt,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
