import time
import math

def secAsMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}min {int(s)}sec'


def since_remaining_time_epoch(since, epoch_part):
    '''
    возвращает пройденное время от старта и оцененное (по доле оставшихся эпох) 
    оставшееся время в минутах и секундах 
        since - время старта
        epoch_part - доля пройденных эпох
    '''
    now = time.time()
    pt = now - since    # passed time

    # estimated remaining time = ((1 - epoch_part) / epoch_part) * pt = 
    # = (1 / epoch_part - 1) * pt = pt / epoch_part - pt
    ert = pt / epoch_part - pt

    return f'passed time = {secAsMinutes(pt)} estimated remaining time = {secAsMinutes(ert)}'


def since_remaining_time(since):
    '''
    возвращает пройденное время от старта в минутах и секундах 
        since - время старта
    '''
    now = time.time()
    pt = now - since    # passed time

    return f'passed time = {secAsMinutes(pt)}', now


def time_report(
    epoch,
    start,
    n_epochs
):
    '''
    формирует строку с временными данными: номер эпохи, процент пройденных эпох, 
    пройденное время, оцененное (по доле оставшихся эпох) оставшееся время в минутах и секундах 
        epoch - текущая эпоха
    '''
    return ' | '.join([
        f'epoch = {epoch}',
        f'epoch percent = {epoch / n_epochs * 100:.2f}%',
        f'{since_remaining_time(start, epoch / n_epochs)}'
    ])
