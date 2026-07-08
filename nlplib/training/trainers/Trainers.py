from transformers import Trainer


class CustomTrainer(Trainer):
    """
    повторяет unsloth.trainer.UnslothTrainer но наследуется от transformers.Trainer а не trl.SFTTrainer.
    _create_unsloth_optimizer - оптимизатор который задает другой learning rate 
    для весов модулей из modules_to_save - параметры с именами типа "... modules_to_save.default.weight" 
    (как правило это входной/выходной эмбеддинг)
    """
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            from unsloth.trainer import _create_unsloth_optimizer
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer