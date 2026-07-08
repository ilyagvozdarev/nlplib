from sklearn import metrics
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np

class Metric(Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"


class metric_stats:

    def __init__(
        self,
        every_n_epoch: int,
        need_print: bool = False,
        description: str =''
    ):
        self.every_n_epoch  = every_n_epoch
        self.need_print  = need_print
        self.values = []
        self.metric_params_accumulated = {}
        self.description = description
        
    def accumulate(self, args):
            for metric_param, accumulated in self.metric_params_accumulated.items():
                accumulated += args[metric_param]

    def reset_metric_params(self):
        for metric in self.metric_params_accumulated:
            if isinstance(self.metric_params_accumulated[metric], list):
                self.metric_params_accumulated[metric] = []
            else:
                self.metric_params_accumulated[metric] = 0
    
    def calculate(self, args):
        self.values.append(self.calculate_(args))
        self.reset_metric_params()
    
    def calculate_(self):
        pass


class loss_stats(metric_stats):

    def __init__(
        self,
        every_n_epoch: int,
        need_print: bool = False
    ):
        super().__init__(
            every_n_epoch, 
            need_print, 
            f'loss average per epoch for {every_n_epoch} epochs'
        )
        self.metric_params_accumulated = {
            'loss_epoch': 0
        }

    def calculate_(self, args):
        loss_avg = self.metric_params_accumulated['loss_epoch'] / self.every_n_epoch
        return loss_avg


class accuracies_stats(metric_stats):
    def __init__(
        self,
        every_n_epoch: int,
        need_print: bool = False,
    ):
        super().__init__(
            every_n_epoch, 
            need_print, 
            f'loss average per epoch for {every_n_epoch} epochs'
        )
        self.metric_params_accumulated = {
            'probas': [],
            'targets': []
        }

    def calculate_(self, args):
        accuracy = metrics.accuracy_score(
            np.array(self.metric_params_accumulated['targets']), 
            np.array(self.metric_params_accumulated['probas']) > args['thresh']
        )
        return accuracy


class metrics_stats:
    metric_names = {
        Metric.LOSS: loss_stats,
        Metric.ACCURACY: accuracies_stats
    }

    def __init__(
        self,
        metrics : list[Metric] = [Metric.LOSS, Metric.ACCURACY, Metric.LOSS],
        every_n_epoch = [100, 100, 20],
        need_print = [True, True, True]
    ):
        self.metrics = [
            self.metric_names[metric](every, _need_print) 
            for metric, every, _need_print 
            in zip(metrics, every_n_epoch, need_print)
        ]

    def calculate_metrics(
        self,
        **kwargs
    ):
        epoch = kwargs['epoch']
        for metric in self.metrics:
            metric.accumulate(kwargs)
            if epoch % metric.every_n_epoch == 0:
                metric.calculate(kwargs)

    def print_metrics_stats(self, epoch):
        for metric in self.metrics:
            if metric.need_print and epoch % metric.every_n_epoch == 0:
                print(f'{metric.description}: {metric.values[-1]}')

        
def plot_metric(
    metric_stats, 
    label
):
    print('x = ', range(1, len(metric_stats.values) + 1), 'y = ', metric_stats.values)
    plt.figure()
    plt.plot(
        range(1, len(metric_stats.values) + 1), 
        metric_stats.values, 
        label=label
    )
    plt.xlabel(metric_stats.description)
    plt.legend()   
