"""
--- NoDuplicatesExceptNegativesBatchSampler ------------------------------------------------------------------------

Motivation:

In MNRL negatives only ever end up in docs_all, i.e. always in the columns of the similarity
matrices; the rows in all four directions are either queries or docs[0] — anchors and positives.
A negative never carries a label, so two identical negatives belonging to different queries
cannot corrupt one. Every other collision is forbidden:

    positive == positive, positive == negative  — a query's positive becomes its negative
    anchor   == anchor    — the positive of one anchor becomes a negative of that same anchor
    anchor   == positive  — in q→d the anchor becomes a negative of itself, in d→q it also
                            produces a wrong label
    anchor   == negative  — in q→d the anchor becomes a negative of itself, in d→q nothing
                            wrong happens

Meanwhile, negatives shared between queries are common for datasets with hard negatives mined
from small corpora. The sampler does not discard conflicting rows, it defers them to later
batches (sampler.py:588), and __len__ is honestly documented as an upper bound. No data is lost,
but the tail batches degenerate into small ones, and for MNRL the batch size is the number of
negatives. That is where the relaxation pays off: fewer deferrals, fuller batches.

The only side effect is that such a negative enters the softmax denominator twice and gets twice
the weight in the gradient. Under hardness_mode its penalty is counted twice as well. That is a
mild re-weighting of one particular negative rather than a wrong label, which is why the
collision is safe.


Implementation details:

Why a small override will not do: get_sample_values returns a flat set of the values of all
columns, and _has_overlap is a plain isdisjoint. Column membership is lost before any check takes
place. And it cannot be expressed with a single set in principle: a negative must conflict with a
positive holding the same text, yet must not conflict with a negative holding that same text —
an asymmetric relation, whereas set membership is symmetric. Two sets are required, which means
__iter__ has to be rewritten (~50 lines, with the deferred linked-list logic preserved verbatim).

Two deliberate limitations: precompute_hashes=True is rejected with an explicit error (the hash
path collapses the columns into a flat array, so membership cannot be recovered), and negative
columns are detected by the "negative" prefix, which covers the triplet and n-tuple formats; for
custom names there is the negative_columns parameter.

NoDuplicates    : [[2, 3, 4], [0], [1]]
ExceptNegatives : [[2, 3, 4], [0, 1]]
rows 0+1 together: True     <- shared negative, allowed
rows 0+2 together: False    <- shared anchor
rows 0+3 together: False    <- shared positive
rows 0+4 together: False    <- positive of one == negative of the other
"""

from collections.abc import Iterator

import numpy as np
import torch

from sentence_transformers.base.sampler import (
    NoDuplicatesBatchSampler,
    _EXCLUDE_DATASET_COLUMNS,
    _sample_value_str,
)


class NoDuplicatesExceptNegativesBatchSampler(NoDuplicatesBatchSampler):
    """
    Like NoDuplicatesBatchSampler, but two rows may share a batch when their only
    common value sits in a negative column.
    """

    def __init__(self, *args, negative_columns: list[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.precompute_hashes:
            raise ValueError(
                "precompute_hashes=True is not supported here: the hash path erases column identity."
            )
        columns = [c for c in self.dataset.column_names if c not in _EXCLUDE_DATASET_COLUMNS]
        if negative_columns is None:
            negative_columns = [c for c in columns if c.startswith("negative")]
        if not negative_columns:
            raise ValueError(f"No negative columns found among {columns}.")
        self.negative_columns = set(negative_columns)

    def _split_values(self, index: int) -> tuple[set[str], set[str]]:
        labeled: set[str] = set()
        negatives: set[str] = set()
        for key, value in self.dataset[index].items():
            if key in _EXCLUDE_DATASET_COLUMNS:
                continue
            target = negatives if key in self.negative_columns else labeled
            target.add(_sample_value_str(value))
        return labeled, negatives

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        num_rows = len(self.dataset)
        if num_rows == 0:
            return

        index_dtype = torch.int32 if num_rows <= np.iinfo(np.int32).max else torch.int64
        remaining_indices = torch.randperm(num_rows, generator=self.generator, dtype=index_dtype).numpy()

        position_dtype = np.int32 if num_rows + 1 <= np.iinfo(np.int32).max else np.int64
        next_positions = np.arange(1, num_rows + 1, dtype=position_dtype)
        next_positions[-1] = -1
        head_position = 0

        while head_position != -1:
            batch_labeled: set[str] = set()
            batch_negatives: set[str] = set()
            batch_indices: list[int] = []
            current_position = head_position
            previous_position = -1
            full_batch = False
            while current_position != -1:
                next_position = int(next_positions[current_position])
                index = int(remaining_indices[current_position])
                labeled, negatives = self._split_values(index)

                # An anchor/positive must not clash with anything already in the batch.
                # A negative must not clash with an anchor/positive, but may repeat a
                # negative that is already there.
                if (
                    not labeled.isdisjoint(batch_labeled)
                    or not labeled.isdisjoint(batch_negatives)
                    or not negatives.isdisjoint(batch_labeled)
                ):
                    # Defer conflicting samples to later batches instead of reordering them.
                    previous_position = current_position
                    current_position = next_position
                    continue

                batch_indices.append(index)
                if previous_position == -1:
                    head_position = next_position
                else:
                    next_positions[previous_position] = next_position

                if len(batch_indices) == self.batch_size:
                    full_batch = True
                    yield batch_indices
                    break

                batch_labeled.update(labeled)
                batch_negatives.update(negatives)
                current_position = next_position

            if not full_batch:
                if not self.drop_last:
                    yield batch_indices