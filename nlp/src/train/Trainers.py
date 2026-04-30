
from transformers import Trainer


class CustomTrainer(Trainer):
    
    # (повторяет unsloth.trainer.UnslothTrainer но наследуется от transformers.Trainer а не trl.SFTTrainer)
    # _create_unsloth_optimizer - оптимизатор который задает другой learning rate для весов модулей из modules_to_save 
    # (имена заканчиваются на "modules_to_save.default.weight", подразумевается что это входной/выходной эмбеддинг)

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


class Trainer_logging(Trainer):

    # с логированием входа и выхода для каждого батча (вызов self.log_step)

    def format_list(self, items, offset=15):
        return ', '.join([f'{{0:<{offset}}}'.format(repr(item)) for item in items])

    def create_info(self, names, items_lists, offset):
        info = '\n'.join([
            self.format_list(items, offset) 
            for items in items_lists
        ])
        return '\n' + '\n'.join(names) + '\n' + info

    def log_step(self, inputs, outputs, log_shapes=False, with_no_attentioned=False):
        tokenizer = self.data_collator.tokenizer
        spec_token_idx = tokenizer.get_vocab()[list(tokenizer.special_tokens_map.values())[-1]]
        # scores = torch.nn.LogSoftmax(dim=-1)(outputs.logits)
        logits = outputs.logits
        preds_batch = logits.argmax(axis=-1)

        # logging.info('\npreds:\n' + repr(preds))
        logging.info('\n\n')

        if log_shapes:
            # logging.info("input_ids.shape = " + str(inputs['input_ids'].shape))
            # logging.info("labels.shape = " + str(inputs['labels'].shape))
            # logging.info("preds.shape = " + str(preds_batch.shape))
            logging.info("attention_mask.shape = " + str(inputs['attention_mask'].shape))

        batch_size = inputs['labels'].shape[0]
        batch_len = inputs['labels'].shape[1] 

        logging.info('batch len = ' + str(batch_len))

        # for name in ['input_ids', 'labels', 'preds_batch']:
        #     print(inputs[name])
        #     logging.info(rf"{name} tokens = \n" + repr(tokenizer.batch_decode(inputs[name])))
        
        for i_batch in range(batch_size):

            logging.info(f'\ni_batch = {i_batch}')

            inputs_outputs = {
                'input_ids': inputs['input_ids'][i_batch],
                'labels': inputs['labels'][i_batch].clone(),
                'preds': preds_batch[i_batch].clone(),
                'attention_mask': inputs['attention_mask'][i_batch]
            }

            items_names = ['input_ids', 'labels', 'preds']
            if not with_no_attentioned:
                for k in items_names:
                    inputs_outputs[k] = inputs_outputs[k][inputs_outputs['attention_mask'] == 1]
            else:
                items_names += ['attention_mask']

            mask_loss = inputs_outputs['labels'] == -100

            logging.info('sample len = ' + str(len(inputs_outputs['labels'])))
            logging.info('masked loss count = ' + str(int(sum(mask_loss))))
            
            tokens_ids_lists = [inputs_outputs[k].tolist() for k in items_names]
            info = self.create_info(items_names, tokens_ids_lists, offset=6)
            logging.info(info)

            inputs_outputs['labels'][mask_loss] = spec_token_idx
            inputs_outputs['preds'][mask_loss] = spec_token_idx

            tokens_lists = [
                [tokenizer.decode(item) for item in inputs_outputs[name]]
                for name in items_names
            ]

            info = self.create_info(items_names, tokens_lists, offset=15)
            logging.info(info)

            logging.info('preds on non-masked = \n' + repr(tokenizer.decode(inputs_outputs['preds'][~mask_loss])))


    # compute_loss с логированием входа и выхода для каждого батча (вызов self.log_step)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            """
            if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            if self.model_accepts_loss_kwargs:
                loss_kwargs = {}
                if num_items_in_batch is not None:
                    loss_kwargs["num_items_in_batch"] = num_items_in_batch
                inputs = {**inputs, **loss_kwargs}
            outputs = model(**inputs)

            self.log_step(inputs, outputs, log_shapes=True)


            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                # User-defined compute_loss function
                if self.compute_loss_func is not None:
                    loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                loss *= self.accelerator.num_processes

            return (loss, outputs) if return_outputs else loss
