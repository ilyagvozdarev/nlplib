import numpy as np
import torch
import torchaudio


def preprocess_waveform(waveform: np.ndarray, orig_sample_rate, new_sample_rate):

    '''
        предобработка waveform:
         - int16, int32 -> float32
         - нормализация к диапазону [-1, 1]:
            int16: [-32768..32767] делим на 32768
            int32: делим на 2147483648
         - добавляем 1-размерность вначале (если в 0 размерности несколько 
           векторов то добавляем только перворму вектору)
         - resample из orig_sample_rate в new_sample_rate
    '''
    if waveform.dtype == "int16":
        waveform = torch.tensor(waveform.astype(np.float32, order="C") / 32768.0).T
    elif waveform.dtype == "int32":
        waveform = torch.tensor(
            waveform.astype(np.float32, order="C") / 2147483648.0
        ).T
    else:
        waveform = torch.tensor(waveform).T

    # добавление размерности
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)

    # ресэмплинг
    if orig_sample_rate != new_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_sample_rate, new_sample_rate)(waveform)

    return waveform