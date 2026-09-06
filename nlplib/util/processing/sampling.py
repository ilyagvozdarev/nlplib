import numpy as np


def get_windows(words, C):
    """
    a sliding window of length C over a given list of words
    """
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C) : i] + words[(i + 1) : (i + C+ 1)]
        yield context_words, center_word
        i += 1


def balanced_gather(datasets: list[np.ndarray], N: int) -> np.ndarray:
    """
    Samples in round-robin style from the list of datasets `datasets`.
    The number of items sampled from each dataset is equivalent to the number
    obtained by round-robin sampling — i.e. when at each step we take one sample from
    the next dataset that has not yet been exhausted, until we have collected N. However,
    instead of O(N) time, this implementation takes O(K) (1 <= K <= number of datasets in `datasets`).
    Idea:
    Note that at each step we can take M samples from every dataset that is not yet exhausted
    (M — the minimum size among the datasets that are not yet exhausted).
    If the accumulated number of samples exceeds N, we stop at the current step
    and take remaining // d from each unexhausted dataset (remaining — the number of samples
    still left to sample, d — the number of unexhausted datasets), and the leftover
    remaining % d are taken one at a time from arbitrary unexhausted datasets.
    """
    K = len(datasets)
    sizes = np.array([len(d) for d in datasets])
    order = np.argsort(sizes)
    sorted_sizes = sizes[order]

    diffs = np.diff(sorted_sizes, prepend=0)
    active_counts = np.arange(K, 0, -1)
    cumulative = np.cumsum(diffs * active_counts)
    counts = np.zeros(K, dtype=int)
    stop_phase = np.searchsorted(cumulative, N, side='left')
    if stop_phase >= K:
        counts = sorted_sizes.copy()
    elif cumulative[stop_phase] == N:
        # Exactly complete through phase stop_phase
        counts[:stop_phase + 1] = sorted_sizes[:stop_phase + 1]
        counts[stop_phase + 1:] = sorted_sizes[stop_phase]
    else:
        # Stop mid-phase
        counts[:stop_phase] = sorted_sizes[:stop_phase]
        base = sorted_sizes[stop_phase - 1] if stop_phase > 0 else 0
        counts[stop_phase:] = base
        remaining = N - counts.sum()
        chunk, rem = divmod(remaining, K - stop_phase)
        counts[stop_phase:] += chunk
        counts[stop_phase:stop_phase + rem] += 1
    result_counts = np.zeros(K, dtype=int)
    result_counts[order] = counts
    return np.concatenate([d[:c] for d, c in zip(datasets, result_counts) if c > 0])