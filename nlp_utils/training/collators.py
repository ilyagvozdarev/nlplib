import torch
from torch import nn


class collator_2seq_pad_lengths(object):

    """
    коллатор который преобразует батч троек [токены последовательности 1, токены последовательности 2, цель]
    в кортеж дополненных последовательностей, их длин до дополнения, целей
    """

    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device

    def __call__(self, batch):
        """    
        Parameters
        ----------
        batch:
            батч троек [токены последовательности 1, токены последовательности 2, цель]

        vocab:
            словарь индексов токенов

        Returns
        -------
        tuple: (
            (padded последовательности 1, длины последовательностей 1 до padding), 
            (padded последовательности 2, длины последовательностей 2 до padding), 
            цели
        )
        """
        def collate(sequences):
            ids_seqs = [
                [self.vocab[token] for token in seq] 
                for seq in sequences
            ]
            lengths_seqs = torch.tensor([len(seq) for seq in sequences])
            padded_seqs = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids_seq) for ids_seq in ids_seqs], 
                batch_first=True, 
                padding_value=0
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
