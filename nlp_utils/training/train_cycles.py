import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
from torch import optim
from torch import nn
import torch.nn.functional as F
import torch
import math
from sklearn import metrics
from .timing import time_report
from .metric_stats import metrics_stats, plot_metric, Metric



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
    thresh = 0.33388096
):
    start = time.time()

    encoder_optimizer = optim.Adam(
        encoder.parameters(), 
        lr=learning_rate
    )
    decoder_attn_optimizer = optim.Adam(
        decoder_attn.parameters(), 
        lr=learning_rate
    )
    criterion = nn.BCEWithLogitsLoss()

    encoder.train()
    decoder_attn.train()

    for epoch in range(1, n_epochs + 1):
        print('epoch = ', epoch)

        loss_epoch = train_epoch(
            train_dataloader, 
            encoder, 
            decoder_attn, 
            encoder_optimizer, 
            decoder_attn_optimizer, 
            criterion
        )

        if epoch % time_every_n_epoch == 0:
            print(time_report(epoch, start, n_epochs))
        
        probas, targets = evaluate(train_dataloader, encoder, decoder_attn)
        metrics_stats_train.calculate_metrics(
            epoch=epoch,
            loss_epoch=loss_epoch,
            probas=probas, targets=targets,
            thresh=thresh
        )

        probas, targets = evaluate(valid_dataloader, encoder, decoder_attn)
        metrics_stats_valid.calculate_metrics(
            epoch=epoch,
            loss_epoch=loss_epoch,
            probas=probas, targets=targets,
            thresh=thresh
        )

        metrics_stats_train.print_metrics_stats(epoch)
        metrics_stats_valid.print_metrics_stats(epoch)

    ## plot metrics ##
    for metric in metrics_stats_train.metrics:
        plot_metric(metric, 'train')
    for metric in metrics_stats_valid.metrics:
        plot_metric(metric, 'valid')

    return None


def evaluate(dataloader, encoder, decoder_attn):

    with torch.no_grad():
        probas = []
        targets_ = []
        for data in dataloader:
            (input_encoder_tensor, lengths1), \
            (input_attn_decoder_tensor, _), \
            targets = data

            encoder_outputs, (encoder_output_hidden, encoder_output_cell) = encoder(
                input_encoder_tensor, 
                lengths1
            )
            decoder_attn_output, _, _, _, _ = decoder_attn(
                encoder_outputs, 
                init_hidden_forward = encoder_output_hidden,
                init_hidden_backward = encoder_output_hidden,
                init_cell_forward = encoder_output_cell,
                init_cell_backward = encoder_output_cell, 
                input = input_attn_decoder_tensor
            )
        
            # print('decoder_attn_output.shape =', decoder_attn_output.shape)
            # print('F.sigmoid(decoder_attn_output) =', F.sigmoid(decoder_attn_output))

            probas.extend(F.sigmoid(decoder_attn_output).squeeze(1).cpu().numpy())
            targets_.extend(targets.squeeze(1).cpu().numpy())

    return probas, targets_


def train_epoch(
    dataloader, 
    encoder, 
    decoder_attn, 
    encoder_optimizer, 
    decoder_attn_optimizer, 
    criterion
):

    total_loss = 0

    for data in dataloader:
        (input_encoder_tensor, lengths1), \
        (input_attn_decoder_tensor, _), \
        targets = data

        encoder_optimizer.zero_grad()
        decoder_attn_optimizer.zero_grad()

        encoder_outputs, (encoder_output_hidden, encoder_output_cell) = encoder(
            input_encoder_tensor, 
            lengths1
        )
        decoder_attn_output, _, _, _, _ = decoder_attn(
            encoder_outputs, 
            init_hidden_forward = encoder_output_hidden,
            init_hidden_backward = encoder_output_hidden,
            init_cell_forward = encoder_output_cell,
            init_cell_backward = encoder_output_cell, 
            input = input_attn_decoder_tensor
        )
        
        # print('decoder_attn_output.shape =', decoder_attn_output.shape)
        # print('targets.shape = ', targets.shape)
        # print(targets.device)

        loss = criterion(
            decoder_attn_output,
            targets.float()
        )
        # print(loss)

        loss.backward()

        encoder_optimizer.step()
        decoder_attn_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epoch_siamese(data, model, optimizer, criterion):
    '''
      Обучает модель на заданных данных (список пар Q1, Q2) одну эпоху, с заданной функцией ошибки и оптимизатором.
      Возвращает среднее значение функции ошибки на батч
    
      Input: 
        data - данные: 
            список пар Q1, Q2 (положительные (одинаковые) примеры) которые могут быть батчами.
            Каждый i-ый Q1 и Q2 - положительные (одинаковые) примеры, все остальные где i!=j - отрицательные друг другу примеры
        model - модель
        criterion, optimizer - функция ошибки, optimizer
        
      Return:
          средняя ошибка на батч
    '''
    
    import torch
    import numpy as np
    
    total_loss = 0
    batches_count = 0
    right_ans_sum = ans_count = 0

    for Q1, Q2, in data:        

        batches_count += 1
        optimizer.zero_grad()
        
        Q1 = torch.tensor(Q1)
        Q2 = torch.tensor(Q2)
        
        output_Q1 = model(Q1)
        output_Q2 = model(Q2)

        loss = criterion(
            output_Q1,
            output_Q2
        )
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / batches_count

    
def train_epoch_classification_multioutput(data, model, optimizer, criterion):
    '''
      Обучает модель на заданных данных (список пар X,Y) одну эпоху, с заданной функцией ошибки и оптимизатором.
      Выводит долю правильных ответов (accuracy).
      Возвращает среднее значение функции ошибки на батч
    
      Input: 
        data - данные: 
            список пар X, Y которые могут быть батчами
            Y - список меток классов (примеры задач - машинный перевод или NER-теггирование)
        model - модель
        criterion, optimizer - функция ошибки, optimizer
        
      Return:
          средняя ошибка на батч
    '''
    
    import torch
    import numpy as np
    
    total_loss = 0
    batches_count = 0
    right_ans_sum = ans_count = 0

    for X1, Y1, in data:        

        batches_count += 1
        optimizer.zero_grad()
        
        X1 = torch.tensor(X1)
        Y1 = torch.tensor(Y1)
        Y1 = Y1.type(torch.LongTensor)
        
        outputs = model(X1)
        outputs = torch.transpose(outputs, 2, 1) 

        preds = np.argmax(outputs.detach().numpy(), axis=1)
        
        mask = Y1 != 35180
        right_ans_sum += np.sum(preds == Y1.detach().numpy())
        ans_count += float(np.sum(mask.detach().numpy()))

        loss = criterion(
            outputs,
            Y1
        )
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print('accuracy = ', right_ans_sum / ans_count)

    return total_loss / batches_count
