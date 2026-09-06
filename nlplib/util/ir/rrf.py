from __future__ import annotations

import csv
import json
import logging
import os

import numpy as np
from tqdm import tqdm

from sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator import ReciprocalRankFusionEvaluator

logger = logging.getLogger(__name__)


class WeightedReciprocalRankFusionEvaluator(ReciprocalRankFusionEvaluator):
    """RRF(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

    Копия __call__ родителя, изменён только блок расчёта rrf_scores.
    """

    def __init__(self, *args, weights: tuple[float, float] = (1.0, 1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def __call__(self, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"ReciprocalRankFusionEvaluator: Evaluating hybrid search on the {self.name} dataset{out_txt}:")
        logger.info(f"Processing {len(self.dense_samples)} samples")

        # Initialize scores
        dense_mrr_scores = []
        dense_ndcg_scores = []
        dense_ap_scores = []

        sparse_mrr_scores = []
        sparse_ndcg_scores = []
        sparse_ap_scores = []

        fusion_mrr_scores = []
        fusion_ndcg_scores = []
        fusion_ap_scores = []

        num_queries = 0
        num_positives = []
        fused_results_list = []

        # Process each pair of samples
        for i, (dense_sample, sparse_sample) in enumerate(
            tqdm(
                zip(self.dense_samples, self.sparse_samples),
                desc="Evaluating",
                disable=not self.show_progress_bar,
                total=len(self.dense_samples),
            )
        ):
            query_id = dense_sample["query_id"]

            # Verify query_id match (redundant since we checked in __init__, but good for safety)
            assert query_id == sparse_sample["query_id"], (
                f"Query ID mismatch: {query_id} != {sparse_sample['query_id']}"
            )

            query = dense_sample["query"]
            positive = dense_sample["positive"]
            if isinstance(positive, str):
                positive = [positive]

            # Get documents from both retrievers
            dense_docs = dense_sample["documents"]
            sparse_docs = sparse_sample["documents"]

            # Calculate base metrics for dense retriever
            dense_is_relevant = [int(sample in positive) for sample in dense_docs]

            # Skip if no relevant documents
            if sum(dense_is_relevant) == 0:
                dense_mrr, dense_ndcg, dense_ap = 0, 0, 0
            else:
                dense_is_relevant += [1] * (len(positive) - sum(dense_is_relevant))
                dense_pred_scores = np.array(range(len(dense_is_relevant), 0, -1))
                dense_mrr, dense_ndcg, dense_ap = self.compute_metrics(dense_is_relevant, dense_pred_scores)

            dense_mrr_scores.append(dense_mrr)
            dense_ndcg_scores.append(dense_ndcg)
            dense_ap_scores.append(dense_ap)

            # Calculate base metrics for sparse retriever
            sparse_is_relevant = [int(sample in positive) for sample in sparse_docs]

            # Skip if no relevant documents
            if sum(sparse_is_relevant) == 0:
                sparse_mrr, sparse_ndcg, sparse_ap = 0, 0, 0
            else:
                sparse_is_relevant += [1] * (len(positive) - sum(sparse_is_relevant))
                sparse_pred_scores = np.array(range(len(sparse_is_relevant), 0, -1))
                sparse_mrr, sparse_ndcg, sparse_ap = self.compute_metrics(sparse_is_relevant, sparse_pred_scores)

            sparse_mrr_scores.append(sparse_mrr)
            sparse_ndcg_scores.append(sparse_ndcg)
            sparse_ap_scores.append(sparse_ap)

            # Create one-based rank maps for each retriever
            dense_ranks = {doc: rank for rank, doc in enumerate(dense_docs, start=1)}
            sparse_ranks = {doc: rank for rank, doc in enumerate(sparse_docs, start=1)}

            # Combine all unique documents
            all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())

            # ↓↓↓ ЕДИНСТВЕННОЕ ОТЛИЧИЕ ОТ РОДИТЕЛЯ ↓↓↓
            w_dense, w_sparse = self.weights
            rrf_scores = {}
            for doc in all_docs:
                score = 0.0
                if doc in dense_ranks:
                    score += w_dense / (self.rrf_k + dense_ranks[doc])
                if doc in sparse_ranks:
                    score += w_sparse / (self.rrf_k + sparse_ranks[doc])
                rrf_scores[doc] = score
            # ↑↑↑ ЕДИНСТВЕННОЕ ОТЛИЧИЕ ОТ РОДИТЕЛЯ ↑↑↑

            fused_docs = sorted(rrf_scores.keys(), key=lambda doc: rrf_scores[doc], reverse=True)

            fusion_is_relevant = [int(sample in positive) for sample in fused_docs]

            num_queries += 1
            num_positives.append(len(positive))

            # Skip if no relevant documents in fusion results
            if sum(fusion_is_relevant) == 0:
                fusion_mrr, fusion_ndcg, fusion_ap = 0, 0, 0
            else:
                fusion_is_relevant += [1] * (len(positive) - sum(fusion_is_relevant))
                fusion_pred_scores = np.array(range(len(fusion_is_relevant), 0, -1))
                fusion_mrr, fusion_ndcg, fusion_ap = self.compute_metrics(fusion_is_relevant, fusion_pred_scores)

            fusion_mrr_scores.append(fusion_mrr)
            fusion_ndcg_scores.append(fusion_ndcg)
            fusion_ap_scores.append(fusion_ap)

            # Store fused results for prediction file if requested
            if self.write_predictions:
                fused_results_list.append(
                    {"query_id": query_id, "query": query, "positive": positive, "documents": fused_docs}
                )

        # Calculate mean scores
        mean_dense_mrr = np.mean(dense_mrr_scores)
        mean_dense_ndcg = np.mean(dense_ndcg_scores)
        mean_dense_ap = np.mean(dense_ap_scores)

        mean_sparse_mrr = np.mean(sparse_mrr_scores)
        mean_sparse_ndcg = np.mean(sparse_ndcg_scores)
        mean_sparse_ap = np.mean(sparse_ap_scores)

        mean_fusion_mrr = np.mean(fusion_mrr_scores)
        mean_fusion_ndcg = np.mean(fusion_ndcg_scores)
        mean_fusion_ap = np.mean(fusion_ap_scores)

        metrics = {
            "dense_map": mean_dense_ap,
            f"dense_mrr@{self.at_k}": mean_dense_mrr,
            f"dense_ndcg@{self.at_k}": mean_dense_ndcg,
            "sparse_map": mean_sparse_ap,
            f"sparse_mrr@{self.at_k}": mean_sparse_mrr,
            f"sparse_ndcg@{self.at_k}": mean_sparse_ndcg,
            "map": mean_fusion_ap,
            f"mrr@{self.at_k}": mean_fusion_mrr,
            f"ndcg@{self.at_k}": mean_fusion_ndcg,
        }

        logger.info(
            f"Queries: {num_queries}\t"
            f"Positives: Min {min(num_positives) if num_positives else 0:.1f}, "
            f"Mean {np.mean(num_positives) if num_positives else 0:.1f}, "
            f"Max {max(num_positives) if num_positives else 0:.1f}"
        )

        logger.info("=" * 75)
        logger.info(
            f"{'Metric':<7} | {'Dense':^8} | {'Sparse':^8} | {'Fusion':^8} | {'Gain vs Dense':^13} | {'Gain vs Sparse':^14} |"
        )
        logger.info("-" * 75)
        logger.info(
            f"MAP     | {mean_dense_ap:>8.2%} | {mean_sparse_ap:>8.2%} | {mean_fusion_ap:>8.2%} | {mean_fusion_ap - mean_dense_ap:>+13.2%} | {mean_fusion_ap - mean_sparse_ap:>+14.2%} |"
        )
        logger.info(
            f"MRR@{self.at_k:<3} | {mean_dense_mrr:>8.2%} | {mean_sparse_mrr:>8.2%} | {mean_fusion_mrr:>8.2%} | {mean_fusion_mrr - mean_dense_mrr:>+13.2%} | {mean_fusion_mrr - mean_sparse_mrr:>+14.2%} |"
        )
        logger.info(
            f"NDCG@{self.at_k:<2} | {mean_dense_ndcg:>8.2%} | {mean_sparse_ndcg:>8.2%} | {mean_fusion_ndcg:>8.2%} | {mean_fusion_ndcg - mean_dense_ndcg:>+13.2%} | {mean_fusion_ndcg - mean_sparse_ndcg:>+14.2%} |"
        )
        logger.info("=" * 75)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if self.write_csv:
                csv_path = os.path.join(output_path, self.csv_file)
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow(
                        [
                            epoch,
                            steps,
                            mean_dense_ap,
                            mean_dense_mrr,
                            mean_dense_ndcg,
                            mean_sparse_ap,
                            mean_sparse_mrr,
                            mean_sparse_ndcg,
                            mean_fusion_ap,
                            mean_fusion_mrr,
                            mean_fusion_ndcg,
                        ]
                    )

            # Write prediction results if requested
            if self.write_predictions and fused_results_list:
                json_path = os.path.join(output_path, self.predictions_file)
                with open(json_path, mode="w", encoding="utf-8") as f:
                    for result in fused_results_list:
                        f.write(json.dumps(result) + "\n")
                logger.info(f"Wrote fused ranking predictions to {json_path}")

        # Prefix metrics with name if provided
        metrics = self.prefix_name_to_metrics(metrics, self.name)

        return metrics

    def get_config_dict(self):
        return {**super().get_config_dict(), "weights": list(self.weights)}