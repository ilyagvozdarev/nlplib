import random
import copy, re
from typing import List, Dict

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from .Dataset import Dataset, content_to_type_and_text


BOT_ROLES = ("assistant", "bot", "gpt")


class ConvDataset(Dataset):
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        tokenize_messages_separately = True,
        add_global_bos: bool = True,
        add_global_eos: bool = True,
        labels_pad_token_id: int = -100,
        convert_content = False
    ):
        '''
            sample_rate: оставляет пример (разговор) с вероятностью sample_rate
            convert_content: если задано то контент каждого сообщения разговора преобразуем в:
                m["content"] = [{"type": "text", "text": m["content"]}]
            max_tokens_count: если длина текущего разговора в токенах превышает max_tokens_count то пропускаем пример
            only_target_loss: 
                если роль сообщения не из ("assistant", "bot", "gpt") и only_target_loss равен True то все labels токенов сообщения 
		        приравниваем labels_pad_token_id (параметр класса = -100) чтобы по ним не считалась ошибка
            add_global_bos, add_global_eos:
                если add_global_bos равен True и первый токен шаблонизированного разговора не равен tokenizer.bos_token_id то добавляем 
                tokenizer.bos_token_id в начало.
		        если add_global_eos равен True то по аналогии с add_global_bos
            
            Notes:
            если предпоследний (как такое может случиться(?)) токен шаблонизированного разговора = tokenizer.eos_token_id то удаляем последний токен

        '''
        super().__init__(conversations, sample_rate, convert_content, tokenizer, max_tokens_count)
        self.only_target_loss = only_target_loss
        self.tokenize_messages_separately = tokenize_messages_separately
        self.labels_pad_token_id = labels_pad_token_id
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.is_printed = False

        self.conversations = []
        
        for conv in tqdm(conversations):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_conversation(conv)
            if tensors is None:
                continue
            self.conversations.append(tensors)


    def get_tokens(self, messages):
        # (если передан список разговоров то оставляет только первый)
        tokens = self.tokenizer.apply_chat_template(
            messages,
            # (tokenizer kwargs) тк токенизироваться может каждое отдельное сообщение,
            # в этом случае спец токены затем добавляются вручную
            add_special_tokens=False,       
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            tokens = tokens[0]

        # Метод может вызываться для каждого сообщения разговора отдельно (а не для всего разговора), поэтому чтобы bos не добавлялся в каждое сообщение 
        # (нам нужен только 1 вначале всего разговора)
        # Причина:
        # видимо некоторые токенизаторы вставляют bos_token_id даже если add_special_tokens=False,
        # поэтому удаляем его так как будем вставлять его далее если add_global_bos == True
        # При этом для eos токена такой обработки нет, видимо потому что eos токен в конце каждого сообщения это нормально

        if tokens[0] == self.tokenizer.bos_token_id:
            print('\ntokens[0] == self.tokenizer.bos_token_id')
            tokens = tokens[1:]
        return tokens
    

    def _validate_tokens(self, input_ids, messages):
        original_input_ids = self.get_tokens(messages)
        assert (
            # input_ids == original_input_ids[: len(input_ids)]
            input_ids == original_input_ids
        ), f"{input_ids} vs {original_input_ids}"

    def _validate_prompt(self, prompt, messages):
        prompt_ = self.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert prompt_ == prompt, f"{prompt_} vs {prompt}"


    def _resolve_special_tokens(self, input_ids, labels):
        if input_ids[0] == self.tokenizer.bos_token_id:
            input_ids = input_ids[1:]
            labels = labels[1:]

        if self.add_global_bos and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            labels.insert(0, self.labels_pad_token_id)

        if input_ids[-2] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
            labels = labels[:-1]
            print('removed eos_token_id since last two tokens was eos_token_id')

        if self.add_global_eos and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        
        return input_ids, labels
    

    def _perform_input_ids_labels_message_separately(self, messages):
        '''
            токенизируем применяя шаблон к каждому сообщению и складываем в один список input_ids, labels,
            если only_target_loss == True то для токенов бота ошибку не считаем (label = self.labels_pad_token_id (-100))

            Notes:
            Поскольку шаблон применяется к каждому сообщению для сообщения с ролью бота возможна ошибка 
            "TemplateError: After the optional system message, conversation roles must alternate user/assistant/user/assistant/..."
        '''
        input_ids, labels = [], []
        for message in messages:
            message_input_ids = self.get_tokens([message])
            message_labels = message_input_ids
            if len(input_ids) + len(message_input_ids) > self.max_tokens_count - 2:
                break

            if message["role"] not in BOT_ROLES and self.only_target_loss:
                message_labels = [self.labels_pad_token_id]*len(message_input_ids)

            input_ids.extend(message_input_ids)
            labels.extend(message_labels)

        self._validate_tokens(input_ids, messages)

        return input_ids, labels


    def _perform_input_ids_labels(self, messages):

        '''
            токенизируем применяя шаблон ко всем сообщениями,
            если only_target_loss == True то для токенов бота (токены контента бота + self.tokenizer.eos_token) ошибку не считаем (label = self.labels_pad_token_id (-100))
            (токены бота находим вручную поиском в шаблонизированном промпте подстроки <контент бота сообщения + eos_token)
        '''

        bot_messages = [msg for msg in messages if msg['role'] in BOT_ROLES]

        # формируем промпт через apply_chat_template (tokenize=True) + decode а не через
        # apply_chat_template (tokenize=False) тк похоже что во втором случае в начало мог бы вставиться bos_token_id 
        # даже если add_special_tokens=False
        prompt_ids = self.get_tokens(messages)
        prompt = self.tokenizer.decode(prompt_ids)
        # все равно проверяем что apply_chat_template (tokenize=True) + decode == apply_chat_template (tokenize=False)
        self._validate_prompt(prompt, messages)

        input = self.tokenizer(prompt, return_offsets_mapping=True)
        input_ids = input['input_ids']

        if (input_ids != prompt_ids):
            print('input_ids и prompt_ids не совпадают!')
            print(input_ids)
            print(prompt_ids)

        if len(input_ids) > self.max_tokens_count - 2:
            return None

        labels = input['input_ids'].copy()

        prompt_ = prompt
        cur_offset = 0

        # формируем bot_repr_spans - индексы токенов начала и конца представлений сообщений бота в исходном шаблонизированном промпте
        bot_repr_spans = []

        for bot_message in bot_messages:
            bot_content_pattern = f'({re.escape(bot_message["content"])}{re.escape(self.tokenizer.eos_token)})'
            try:
                span_l, span_r = re.search(bot_content_pattern, prompt_).span(1)
            except:
                print('bot_content_pattern: ', bot_content_pattern)
                print('\n\nprompt_: ', prompt_)
            span_l += cur_offset
            span_r += cur_offset

            # bounds - индексы токенов начала и конца представления текущего сообщения бота в исходном шаблонизированном промпте
            # предполагается что начало и конец представления сообщения совпадает с границами токенов
            bounds = []
            for i, (left, right) in enumerate(input['offset_mapping']):
                if left == span_l or right == span_r:
                    bounds.append(i)
            # print('span_l = ', span_l, 'span_r', span_r)
            # print('offsets = ', input['offset_mapping'])
            # print('bounds = ', bounds)
            if len(bounds) != 2:
                print('len(bounds) != 2')
                print(prompt_)
            token_id_l, token_id_r = bounds

            bot_repr_spans.append((token_id_l, token_id_r))
            prompt_ = prompt_[span_r:]
            cur_offset += span_r

        # для токенов бота label = self.labels_pad_token_id (-100)
        mask_ids = list(chain(*[range(l, r+1) for l, r in bot_repr_spans]))
        mask = np.full(len(labels), True)
        mask[mask_ids] = False

        labels = np.array(labels)
        labels[mask] = -100
        labels = list(labels)

        self._validate_tokens(input_ids, messages)

        return input_ids, labels


    def convert_conversation(self, conversation):

        messages = copy.deepcopy(conversation["messages"])

		# если convert_content контент каждого сообщения разговора преобразуем в:
		#   m["content"] = [{"type": "text", "text": m["content"]}]
        if self.convert_content:
            messages = content_to_type_and_text(messages)

        perform_input_ids_labels_method = [
            self._perform_input_ids_labels,
            self._perform_input_ids_labels_message_separately
        ][self.tokenize_messages_separately]

        input_ids, labels = perform_input_ids_labels_method(messages)
        input_ids, labels = self._resolve_special_tokens(self, input_ids, labels)

        if not self.is_printed:
            print(input_ids)
            print(labels)
            print(
                "Full prompt:",
                self.tokenizer.decode(input_ids, skip_special_tokens=False),
            )
            self.is_printed = True

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.new_ones(input_ids.size())
        assert (
            input_ids.size(0)
            == labels.size(0)
            == attention_mask.size(0)
            <= self.max_tokens_count
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
