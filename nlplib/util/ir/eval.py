def eval_pipeline(model_ce, dense_pred, qrels_eval, passages,
                  top_k_retrieval=30, ks=(1, 3, 5, 10), mrr_at_k=10,
                  map_at_k=100, batch_size=64, show_progress_bar=False):
    """
    End-to-end оценка: top-N ретривера -> реранкинг кросс-энкодером -> top-k

    Метрики считаются по формулам InformationRetrievalEvaluator, поэтому результат напрямую сопоставим.
    """
    all_pairs = []
    meta = []  # (cand_ids, gold) либо None, если кандидатов нет

    for sample in dense_pred:
        gold = qrels_eval[sample["query_id"]]
        gold = set(gold) if isinstance(gold, (list, set, tuple)) else {gold}
        if not gold:
            continue  # IR-эвалуатор исключает такие запросы из усреднения

        cand_ids = [d["corpus_id"] for d in sample["results"]][:top_k_retrieval]
        if not cand_ids:
            meta.append(None)
            continue

        meta.append((cand_ids, gold))
        all_pairs.extend([sample["query"], passages[cid]] for cid in cand_ids)

    all_scores = model_ce.predict(
        all_pairs, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    ) if all_pairs else np.array([])

    metrics = defaultdict(list)
    offset = 0

    for item in meta:
        if item is None:
            for k in ks:
                for name in ("accuracy", "precision", "recall", "ndcg"):
                    metrics[f"{name}@{k}"].append(0.0)
            metrics[f"mrr@{mrr_at_k}"].append(0.0)
            metrics[f"map@{map_at_k}"].append(0.0)
            continue

        cand_ids, gold = item
        scores = all_scores[offset:offset + len(cand_ids)]
        offset += len(cand_ids)

        # ties разрываются по возрастанию corpus_id — как в IR-эвалуаторе
        order = sorted(range(len(cand_ids)),
                       key=lambda i: (-float(scores[i]), cand_ids[i]))
        rel = [1 if cand_ids[i] in gold else 0 for i in order]

        for k in ks:
            top = rel[:k]
            hits = sum(top)
            metrics[f"accuracy@{k}"].append(1.0 if hits else 0.0)
            metrics[f"precision@{k}"].append(hits / k)
            metrics[f"recall@{k}"].append(hits / len(gold))

            dcg = sum(r / np.log2(i + 2) for i, r in enumerate(top))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), k)))
            metrics[f"ndcg@{k}"].append(dcg / idcg if idcg else 0.0)

        metrics[f"mrr@{mrr_at_k}"].append(
            next((1 / (i + 1) for i, r in enumerate(rel[:mrr_at_k]) if r), 0.0)
        )

        num_correct, sum_prec = 0, 0.0
        for rank, r in enumerate(rel[:map_at_k]):
            if r:
                num_correct += 1
                sum_prec += num_correct / (rank + 1)
        metrics[f"map@{map_at_k}"].append(sum_prec / min(map_at_k, len(gold)))

    return {name: float(np.mean(vals)) for name, vals in sorted(metrics.items())}