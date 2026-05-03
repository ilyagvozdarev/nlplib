from utils_additional import timeSince


def train_n_epoch_with_print_plot_loss(data, model, n_epochs, criterion, optimizer, train_epoch_method, scheduler=None, 
          print_every_n_epoch = 1, 
          plot_every_n_epoch = 1):
    '''
      Обучает модель на заданных данных (список пар X,Y) заданное количество эпох, с заданной функцией ошибки
      и оптимизатором.
      Есть возможность каждые n эпох выводить среднюю по эпохам значение ошибки и
      строить график средней по эпохам ошибки для каждых n эпох, номер текущей эпохи
    
      Input: 
        data - данные (список пар X,Y которые могут быть батчами)
        model - модель
        n_epochs - количество эпох обучения
        criterion, optimizer - функция ошибки, optimizer
        print_every_n_epoch - через каждые сколько эпох нужно печатать среднюю по эпохам значение ошибки
        plot_every_n_epoch - через каждые сколько эпох нужно строить график средней по эпохам ошибки
    '''

    def showPlot(points):
    
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        import matplotlib.ticker as ticker
        import numpy as np

        plt.figure()
        fig, ax = plt.subplots()

        loc = ticker.MultipleLocator(base=0.05)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        plt.show()
        
    
    import time
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    
    for epoch in range(1, n_epochs + 1):
        
        loss = train_epoch_method(data, model, optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every_n_epoch == 0:
            print_loss_avg = print_loss_total / print_every_n_epoch
            print_loss_total = 0
            print('%s (epoch=%d/%d) loss=%.4f' % (timeSince(start), epoch, n_epochs, print_loss_avg))

        if epoch % plot_every_n_epoch == 0:
            plot_loss_avg = plot_loss_total / plot_every_n_epoch
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if scheduler:
            scheduler.step()

            
    showPlot(plot_losses)
    


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
