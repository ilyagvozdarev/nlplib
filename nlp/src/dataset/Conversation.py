import json
# Taken from https://github.com/IlyaGusev/rulm and modified

DEFAULT_MESSAGE_TEMPLATE = "{content}\n"
DEFAULT_SYSTEM_PROMPT = ""

'''
    класс с методами обработки разговора

    для каждой роли задается свой шаблон сообщения

'''

class Conversation:
    def __init__(
        self,
        system_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        user_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template_incomplete: str = DEFAULT_MESSAGE_TEMPLATE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        system_role: str = "system",
        user_role: str = "user",
        bot_role: str = "bot",
        global_prefix: str = '',
        bot_suffix: str = "<s>bot",
        add_special_tokens: bool = False,
        eos_token: str =''
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.system_role = system_role
        self.user_role = user_role
        self.bot_role = bot_role
        self.global_prefix = global_prefix
        self.bot_suffix = bot_suffix
        self.bot_message_template_incomplete = bot_message_template_incomplete
        self.add_special_tokens = add_special_tokens
        self.messages = []

        if system_prompt is not None and len(system_prompt) > 0:
            self.messages.append({
                "role": self.system_role,
                "content": system_prompt
            })

    def iter_messages(self):
        for message in self.messages:
            yield self.apply_template_message(message), message["role"]


    @classmethod
    def from_template(cls, file_name):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template
        )
    

    def add_system_message(self, message):
        if len(self.messages) == 0:
            self.messages.append({
                "role": self.system_role,
                "content": message
            })
        else:
            if self.messages[0]["role"] == self.system_role:
                self.messages[0]["content"] = message
            else:
                self.messages = [{
                    "role": self.system_role,
                    "content": message
                }] + self.messages


    def add_user_message(self, message):
        self.messages.append({
            "role": self.user_role,
            "content": message
        })


    def add_bot_message(self, message):
        self.messages.append({
            "role": self.bot_role,
            "content": message
        })


    def apply_template_message(self, message, incomplete_last_bot_message=False):
        # применяем к сообщению шабон в соответствии с ролью
        if message["role"] == self.system_role:
            return self.system_message_template.format(**message)
        if message["role"] == self.user_role:
            return self.user_message_template.format(**message)
        if message["role"] == self.bot_role:
            if incomplete_last_bot_message:
                return self.bot_message_template_incomplete.format(**message)
            return self.bot_message_template.format(**message)

        raise Exception('Unknown role')
    

    def count_tokens(self, tokenizer, current_messages):
        # количество токенов = суммарное количество токенов в токенизированных шаблонизированных сообщениях 
        final_text = ""
        for message in current_messages:
            final_text += self.apply_template_message(message)
        tokens = tokenizer([final_text])["input_ids"][0]
        return len(tokens)


    def shrink(self, tokenizer, messages, max_tokens):
        # сокрущаем количество сообщений пока количество токенов разговора не станет меньше max_tokens.
        # если первое сообщение system - то оставляем его, дальше удаляем по 2 сообщения (turn user-bot) 
        system_message = []
        other_messages = messages
        if messages[0]["role"] == self.system_role:
            system_message = [messages[0]]
            other_messages = messages[1:]
        while self.count_tokens(tokenizer, system_message + other_messages) > max_tokens:
            other_messages = other_messages[2:]
        return system_message + other_messages


    def get_prompt(self, tokenizer=None, max_tokens: int = None, add_suffix_bot: bool = True, incomplete_last_bot_message: bool = False):
        '''
            формирует шаблонизированный промпт

                incomplete_last_bot_message: 
                    применять ли шаблон self.bot_message_template_incomplete для последнего сообщения с ролью бота
                add_suffix_bot: 
                    добавлять ли в конец промпта self.bot_suffix если последнее сообщение с ролью бота
            
            Notes:
                в шаблонизированный промпт вначало добавляется self.global_prefix
        '''

        messages = self.messages
        if max_tokens is not None:
            assert tokenizer is not None
            messages = self.shrink(tokenizer, messages, max_tokens)

        final_text = self.global_prefix
        for i, message in enumerate(messages):
            if i == len(messages) - 1 and incomplete_last_bot_message and message['role'] == self.bot_role:
                final_text += self.apply_template_message(message, incomplete_last_bot_message=True)
            else:
                final_text += self.apply_template_message(message)

        if add_suffix_bot and messages[-1]['role'] != self.bot_role:
            final_text += self.bot_suffix
            return final_text

        return final_text


    def expand(self, messages, role_mapping = None):
        '''
            добавляем новые сообщения к текущим (self.messages).
            если первое добавляемое сообщение system, то удаляем все текущие сообщения
            
                role_mapping: маппинг ролей из messages
        '''

        if not role_mapping:
            role_mapping = dict()

        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            self.messages.append({
                "role": role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })
