from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def bleu4(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return 0.0
    precisions: list[float] = []
    for n in range(1, 5):
        pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(max(len(pred) - n + 1, 0)))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(max(len(ref) - n + 1, 0)))
        matches = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((matches + 1.0) / (total + 1.0))
    brevity = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / 4)


def score_text(prediction: str, reference: str, metrics: list[str]) -> dict[str, float]:
    functions: dict[str, Callable[[str, str], float]] = {
        "exact_match": exact_match,
        "normalized_exact_match": normalized_exact_match,
        "token_f1": token_f1,
        "rouge_l": rouge_l_f1,
        "bleu4": bleu4,
    }
    scores: dict[str, float] = {}
    for metric in metrics:
        try:
            scores[metric] = float(functions[metric](prediction, reference))
        except KeyError as exc:
            raise ValueError(f"Unknown text metric: {metric}") from exc
    return scores
