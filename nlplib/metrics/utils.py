from typing import Any, Sequence
 
import numpy as np
 
# (label_value, pred_value) for each confusion-matrix bucket
RATE_DEFINITIONS: dict[str, tuple[int, int]] = {
    'FP': (0, 1),
    'FN': (1, 0),
    'TP': (1, 1),
    'TN': (0, 0),
}
 
 
def create_rates_binary(
    labels: Sequence[int],
    preds: Sequence[int],
    examples: Sequence[Any],
) -> dict[str, list[Any]]:
    """
    Split `examples` into FP/FN/TP/TN buckets based on `labels` vs `preds`.
 
    Returns a dict with keys 'FP', 'FN', 'TP', 'TN', each mapping to the
    list of examples that fall into that bucket (original order preserved).
    """
    if not (len(labels) == len(preds) == len(examples)):
        raise ValueError(
            f"labels ({len(labels)}), preds ({len(preds)}) and examples "
            f"({len(examples)}) must have the same length"
        )
 
    labels_ = np.asarray(labels)
    preds_ = np.asarray(preds)
 
    res_rates: dict[str, list[Any]] = {}
    for rate_name, (label_val, pred_val) in RATE_DEFINITIONS.items():
        mask = (labels_ == label_val) & (preds_ == pred_val)
        res_rates[rate_name] = [examples[i] for i in mask.nonzero()[0]]
 
    return res_rates
 
 
def _correct_ids(labels: Sequence[int], preds: Sequence[int]) -> set[int]:
    """
    Indices where the prediction matches the label (i.e. TP or TN).
    """
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    return set((labels_arr == preds_arr).nonzero()[0].tolist())
 
 
def create_rates(
    labels: Sequence[Sequence[int]],
    preds: Sequence[Sequence[int]],
    examples: Sequence[Any],
) -> dict[str, Any]:
    """
    Per-violation version of `create_rates_binary`, plus a 'full_true' bucket
    of examples predicted correctly across *every* violation.
 
    Returns
    -------
        {
            'per_violation': {
                0: {'FP': [...], 'FN': [...], 'TP': [...], 'TN': [...]},
                1: {...},
                ...
            },
            'full_true': [...],
        }
    """
    if not labels:
        return {'per_violation': {}, 'full_true': []}
 
    if len(labels) != len(preds):
        raise ValueError(
            f"labels ({len(labels)}) and preds ({len(preds)}) must have the same length"
        )
 
    per_violation: dict[int, dict[str, list[Any]]] = {}
    correct_ids_per_violation: list[set[int]] = []
 
    for viol, (viol_labels, viol_preds) in enumerate(zip(labels, preds)):
        per_violation[viol] = create_rates_binary(viol_labels, viol_preds, examples)
        correct_ids_per_violation.append(_correct_ids(viol_labels, viol_preds))
 
    full_true_ids = set.intersection(*correct_ids_per_violation)
    full_true = [example for i, example in enumerate(examples) if i in full_true_ids]
 
    return {'per_violation': per_violation, 'full_true': full_true}
 
 
if __name__ == '__main__':
    # quick sanity check
    labels_ = [[0, 1, 1, 0], [1, 1, 0, 0]]
    preds_ = [[0, 1, 0, 0], [1, 0, 0, 1]]
    examples_ = ['a', 'b', 'c', 'd']
 
    result = create_rates(labels_, preds_, examples_)
    print(result)