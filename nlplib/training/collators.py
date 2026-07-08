import torch
from torch import nn


class collator_2seq_pad_lengths(object):
    """
    Collator that transforms a batch of triples [tokens of sequence 1, tokens of sequence 2, target]
    into a tuple of padded sequences, their pre-padding lengths, and targets.

    Important (for the calling code): sequences are NOT sorted by length.
    When calling nn.utils.rnn.pack_padded_sequence on this collator's output,
    you must pass enforce_sorted=False - otherwise PyTorch will raise an error
    on the first batch whose lengths aren't in decreasing order. Sorting both
    sequences as pairs here isn't possible: seq1 and seq2 generally have
    different length distributions, and sorting by one would break the sort order of the other.
    """

    def __init__(self, vocab, device, unk_token='<unk>', pad_idx=0):
        """
        Parameters
        ----------
        vocab:
            token index dictionary
        device:
            device to which the padded sequences and targets are moved
        unk_token:
            token used for words missing from the vocabulary (OOV). If this key
            is not present in vocab, pad_idx is used instead (in that case OOV
            words will be indistinguishable from padding for the model - pass
            your own unk_token/extend vocab if you need to avoid this)
        pad_idx:
            padding index. Must be reserved in vocab specifically for padding
            and must not coincide with the index of any real token
        """
        self.vocab = vocab
        self.device = device
        self.pad_idx = pad_idx
        self.unk_idx = vocab.get(unk_token, pad_idx)

    def __call__(self, batch):
        """    
        Parameters
        ----------
        batch:
            batch of triples [tokens of sequence 1, tokens of sequence 2, target]

        Returns
        -------
        tuple: (
            (padded sequence 1, sequence 1 lengths before padding), 
            (padded sequence 2, sequence 2 lengths before padding), 
            targets
        )
        Lengths are intentionally kept on CPU (not moved to device) - 
        this matches what nn.utils.rnn.pack_padded_sequence expects.
        """
        def collate(sequences):
            ids_seqs = []
            for seq in sequences:
                if len(seq) == 0:
                    raise ValueError(
                        "Empty sequence detected (0 tokens): "
                        "pack_padded_sequence does not support zero length. "
                        "Filter out such examples before collation."
                    )
                ids_seqs.append(
                    [self.vocab.get(token, self.unk_idx) for token in seq]
                )

            lengths_seqs = torch.tensor(
                [len(ids_seq) for ids_seq in ids_seqs], dtype=torch.long
            )
            padded_seqs = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids_seq, dtype=torch.long) for ids_seq in ids_seqs],
                batch_first=True,
                padding_value=self.pad_idx
            ).to(self.device)
            return lengths_seqs, padded_seqs

        seqs1, seqs2, targets = list(zip(*batch))
        targets = torch.tensor(targets).to(self.device)

        lengths_seqs1, padded_seqs1 = collate(seqs1)
        lengths_seqs2, padded_seqs2 = collate(seqs2)

        return (
            (padded_seqs1, lengths_seqs1), 
            (padded_seqs2, lengths_seqs2), 
            targets
        )       
