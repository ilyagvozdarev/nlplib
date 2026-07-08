import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import torch
from torch import optim
from torch import nn

from .timing import time_report
from .metric_stats import plot_metric


# Decision threshold tuned on a held-out set; kept as a default so callers
# don't have to know the value, but can always override it.
DEFAULT_THRESH = 0.33388096

# Index of the padding label in the target vocabulary. Used to mask out
# padded positions when computing token-level accuracy.
PAD_LABEL_ID = 35180


def _get_device(device=None):
    if device is not None:
        return device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_encoder_decoder_attn(
    train_dataloader,
    valid_dataloader,
    encoder,
    decoder_attn,
    n_epochs,
    metrics_stats_train,
    metrics_stats_valid,
    learning_rate=0.001,
    time_every_n_epoch=100,
    thresh=DEFAULT_THRESH,
    device=None,
):
    """
    Trains encoder + decoder_attn (bidirectional attention decoder) for
    the given number of epochs, collecting metrics separately for train
    and valid.
    """
    device = _get_device(device)
    encoder.to(device)
    decoder_attn.to(device)

    start = time.time()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_attn_optimizer = optim.Adam(decoder_attn.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs + 1):
        print('epoch = ', epoch)

        encoder.train()
        decoder_attn.train()

        train_loss, train_probas, train_targets = train_epoch(
            train_dataloader,
            encoder,
            decoder_attn,
            encoder_optimizer,
            decoder_attn_optimizer,
            criterion,
            device,
        )

        encoder.eval()
        decoder_attn.eval()
        valid_loss, valid_probas, valid_targets = evaluate(
            valid_dataloader, encoder, decoder_attn, criterion, device
        )

        if epoch % time_every_n_epoch == 0:
            print(time_report(epoch, start, n_epochs))

        metrics_stats_train.calculate_metrics(
            epoch=epoch,
            loss_epoch=train_loss,
            probas=train_probas, targets=train_targets,
            thresh=thresh
        )
        metrics_stats_valid.calculate_metrics(
            epoch=epoch,
            loss_epoch=valid_loss,
            probas=valid_probas, targets=valid_targets,
            thresh=thresh
        )

        metrics_stats_train.print_metrics_stats(epoch)
        metrics_stats_valid.print_metrics_stats(epoch)

    for metric in metrics_stats_train.metrics:
        plot_metric(metric, 'train')
    for metric in metrics_stats_valid.metrics:
        plot_metric(metric, 'valid')


def evaluate(dataloader, encoder, decoder_attn, criterion, device=None):
    """
    Runs encoder/decoder_attn over the dataloader without computing
    gradients. Returns (average loss, probabilities, true labels).

    Models must be switched to the appropriate mode (train()/eval())
    by the caller beforehand — this function does not change the mode.
    """
    device = _get_device(device)

    total_loss = 0.0
    probas = []
    targets_ = []

    with torch.no_grad():
        for data in dataloader:
            (input_encoder_tensor, lengths1), \
            (input_attn_decoder_tensor, _), \
            targets = data

            input_encoder_tensor = input_encoder_tensor.to(device)
            input_attn_decoder_tensor = input_attn_decoder_tensor.to(device)
            targets = targets.to(device)

            encoder_outputs, (encoder_output_hidden, encoder_output_cell) = encoder(
                input_encoder_tensor,
                lengths1
            )
            decoder_attn_output, _, _, _, _ = decoder_attn(
                encoder_outputs,
                init_hidden_forward=encoder_output_hidden,
                init_hidden_backward=encoder_output_hidden,
                init_cell_forward=encoder_output_cell,
                init_cell_backward=encoder_output_cell,
                input=input_attn_decoder_tensor
            )

            loss = criterion(decoder_attn_output, targets.float())
            total_loss += loss.item()

            probas.extend(torch.sigmoid(decoder_attn_output).squeeze(1).cpu().numpy())
            targets_.extend(targets.squeeze(1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, probas, targets_


def train_epoch(
    dataloader,
    encoder,
    decoder_attn,
    encoder_optimizer,
    decoder_attn_optimizer,
    criterion,
    device=None,
):
    """
    Trains encoder/decoder_attn for one epoch.

    In addition to the average loss, it collects predictions/targets
    along the way from the same forward passes used for training — this
    saves a separate full pass over train_dataloader just for metrics,
    at the cost that predictions from the start of the epoch come from
    slightly less-trained weights than those from the end (a common and
    generally acceptable trade-off for learning curves).
    """
    device = _get_device(device)

    total_loss = 0.0
    probas = []
    targets_ = []

    for data in dataloader:
        (input_encoder_tensor, lengths1), \
        (input_attn_decoder_tensor, _), \
        targets = data

        input_encoder_tensor = input_encoder_tensor.to(device)
        input_attn_decoder_tensor = input_attn_decoder_tensor.to(device)
        targets = targets.to(device)

        encoder_optimizer.zero_grad()
        decoder_attn_optimizer.zero_grad()

        encoder_outputs, (encoder_output_hidden, encoder_output_cell) = encoder(
            input_encoder_tensor,
            lengths1
        )
        decoder_attn_output, _, _, _, _ = decoder_attn(
            encoder_outputs,
            init_hidden_forward=encoder_output_hidden,
            init_hidden_backward=encoder_output_hidden,
            init_cell_forward=encoder_output_cell,
            init_cell_backward=encoder_output_cell,
            input=input_attn_decoder_tensor
        )

        loss = criterion(decoder_attn_output, targets.float())
        loss.backward()

        encoder_optimizer.step()
        decoder_attn_optimizer.step()

        total_loss += loss.item()

        probas.extend(torch.sigmoid(decoder_attn_output).detach().squeeze(1).cpu().numpy())
        targets_.extend(targets.squeeze(1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, probas, targets_


def train_epoch_siamese(data, model, optimizer, criterion, device=None):
    """
    Trains the model on the given data (a list of Q1, Q2 pairs) for
    one epoch, using the given loss function and optimizer.
    Returns the average loss per batch.

    Parameters
    ----------
    data - the data:
        a list of Q1, Q2 pairs (positive/matching examples), which
        may be batches. Each i-th Q1 and Q2 is a positive
        (matching) pair; all others, where i != j, are negative
        examples relative to each other.
    model - the model
    optimizer, criterion - optimizer, loss function

    Returns
    -------
    average loss per batch
    """
    device = _get_device(device)

    total_loss = 0.0
    batches_count = 0

    for Q1, Q2 in data:
        batches_count += 1
        optimizer.zero_grad()

        Q1 = torch.as_tensor(Q1, device=device)
        Q2 = torch.as_tensor(Q2, device=device)

        output_Q1 = model(Q1)
        output_Q2 = model(Q2)

        loss = criterion(output_Q1, output_Q2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / batches_count


def train_epoch_classification_multioutput(data, model, optimizer, criterion, device=None):
    """
    Trains the model on the given data (a list of X, Y pairs) for
    one epoch, using the given loss function and optimizer.
    Prints the fraction of correct answers (accuracy), counting only
    real (non-padding) tokens.
    Returns the average loss per batch.

    Parameters
    ----------
        data - the data:
            a list of X, Y pairs, which may be batches.
            Y - a list of class labels (example tasks - machine
            translation or NER tagging).
        model - the model
        optimizer, criterion - optimizer, loss function

    Returns
    -------
    average loss per batch
    """
    device = _get_device(device)

    total_loss = 0.0
    batches_count = 0
    right_ans_sum = 0
    ans_count = 0

    for X1, Y1 in data:
        batches_count += 1
        optimizer.zero_grad()

        X1 = torch.as_tensor(X1, device=device)
        Y1 = torch.as_tensor(Y1, device=device).long()

        outputs = model(X1)
        outputs = torch.transpose(outputs, 2, 1)

        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y1_np = Y1.detach().cpu().numpy()
        mask = y1_np != PAD_LABEL_ID

        # Only count correctness on non-padding positions, matching the
        # denominator below, otherwise accuracy is inflated by "free"
        # correct guesses on padding.
        right_ans_sum += np.sum((preds == y1_np) & mask)
        ans_count += float(np.sum(mask))

        loss = criterion(outputs, Y1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('accuracy = ', right_ans_sum / ans_count)
    return total_loss / batches_count