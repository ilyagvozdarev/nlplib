import random
import copy, re
from typing import List, Dict
from tqdm import tqdm
from itertools import chain

import numpy as np
import torch
from transformers import AutoTokenizer


BOT_ROLES = ("assistant", "bot", "gpt")


import copy
from typing import List, Dict
from transformers import AutoTokenizer

from torch.utils.data import Dataset


def content_to_type_and_text(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        m["content"] = [{"type": "text", "text": m["content"]}]
    return messages


class Dataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        original_records: List[Dict],
        sample_rate: float = 1.0,
        convert_content = False
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.convert_content = convert_content
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


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
        Parameters
        ----------
        sample_rate: keeps the example (conversation) with probability sample_rate
        convert_content: if set, converts the content of every message in the conversation to:
            m["content"] = [{"type": "text", "text": m["content"]}]
        max_tokens_count: if the length of the current conversation in tokens exceeds max_tokens_count, 
            skip the example
        only_target_loss: 
            if the message's role is not one of ("assistant", "bot", "gpt") and only_target_loss is True, 
            then all labels for the message's tokens are set to labels_pad_token_id (class parameter = -100) 
            so that loss is not computed on them
        add_global_bos, add_global_eos:
            if add_global_bos is True and the first token of the templated conversation is not equal to 
            tokenizer.bos_token_id, then tokenizer.bos_token_id is added at the beginning.
            if add_global_eos is True, the analogous logic is applied
            
        Notes
        ----------
        if the second-to-last token of the templated conversation equals tokenizer.eos_token_id, 
        then the last token is removed
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
        '''
        if a list of conversations is specified take only the first one
        '''
        tokens = self.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,       
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            tokens = tokens[0]
        if tokens[0] == self.tokenizer.bos_token_id:
            print('\ntokens[0] == self.tokenizer.bos_token_id')
            tokens = tokens[1:]
        return tokens
    
    def _validate_tokens(self, input_ids, messages):
        original_input_ids = self.get_tokens(messages)
        assert (
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
        tokenize by applying the template to each message and append the results into 
        a single list of input_ids, labels;
        if only_target_loss == True, loss is not computed for the bot's tokens 
        (label = self.labels_pad_token_id (-100))

        Notes
        -----
        Since the template is applied to each message individually, applying it to a 
        message with the bot role may raise a "TemplateError: After the optional system 
        message, conversation roles must alternate user/assistant/user/assistant/..." error
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
        tokenize by applying the template to all messages at once.
        if only_target_loss == True, loss is not computed for the bot's tokens - bot content 
        tokens + self.tokenizer.eos_token (label = self.labels_pad_token_id (-100)).
        bot tokens are located manually by searching the templated prompt for the substring 
        <bot message content> + eos_token
        '''
        bot_messages = [msg for msg in messages if msg['role'] in BOT_ROLES]
        prompt_ids = self.get_tokens(messages)
        prompt = self.tokenizer.decode(prompt_ids)

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

        # form bot_repr_spans — the start and end token indices of the bot messages' representations within 
        # the original templated prompt
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

            # bounds - the indices of the beginning and end tokens of the current bot message 
            # in the original templated prompt
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

        # for bot tokens label = self.labels_pad_token_id (-100)
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
