import os
from collections import defaultdict
import numpy as np


def create_rates(labels, preds, examples):

    full_true_utts = []
    rates = [[0, 1], [1, 0], [1, 1], [0, 0]]

    res_rates = defaultdict(dict)

    for viol in range(len(labels)):
        rates_ids = [
            (np.array(labels[viol]) == i) & (np.array(preds[viol]) == j)
            for i, j in rates
        ]
        FP_ids, FN_ids, TP_ids, TN_ids = [rate_mask.nonzero()[0].tolist() for rate_mask in rates_ids]
        full_true_utts.append(TP_ids + TN_ids)

        for rate_ids, rate_name in [(FP_ids, 'FP'), (FN_ids, 'FN'), (TP_ids, 'TP'), (TN_ids, 'TN')]:
            res_rates[viol][rate_name] = [example for i, example in enumerate(examples) if i in rate_ids]

    # диалоги у которых предсказания по всем ошибкам правильные
    full_true_ids = set(full_true_utts[0]).intersection(*full_true_utts[1:])
    res_rates['full_true'] = [example for i, example in enumerate(examples) if i in full_true_ids]

    return res_rates



def create_rates_binary(labels, preds, examples):

    rates = [[0, 1], [1, 0], [1, 1], [0, 0]]

    res_rates = defaultdict(dict)

    rates_ids = [
        (np.array(labels) == i) & (np.array(preds) == j)
        for i, j in rates
    ]
    FP_ids, FN_ids, TP_ids, TN_ids = [rate_mask.nonzero()[0].tolist() for rate_mask in rates_ids]

    for rate_ids, rate_name in [(FP_ids, 'FP'), (FN_ids, 'FN'), (TP_ids, 'TP'), (TN_ids, 'TN')]:
        res_rates[rate_name] = [example for i, example in enumerate(examples) if i in rate_ids]

    return res_rates


