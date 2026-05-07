
import json
from typing import List, Dict, Union, Any


MSG_TEMPL = """
    Говорящий: {spk}
    Номер сообщения: {n}
    Текст сообщения: {msg}
"""

MSG_TEMPL_2 = """
    Сообщение {n}. {spk}: {msg}"""

MSG_TEMPL_3 = """
    Сообщение {n}:
    {spk}: {msg}
"""


def chat_to_text(chat, msg_template=MSG_TEMPL, field='text'):
    chat = ''
    for i, msg in enumerate(utt):
        # todo перенести в обработку 
        # text = msg[field]
        # if text[0] != '«' and text[-1] != '»':
        #     text = '«' + text + '»'
        chat += msg_template.format(spk=msg['spk'], msg=msg[field], n=i+1)
    return chat.rstrip()




def json_to_markdown_table(
    data: List[Dict[str, Any]],
    fix_format = False,
    sep_row_full = False
) -> str:
    '''
        Преобразует JSON (словарь) в Markdown-таблицу.
        - все встреченные ключи первого уровня преобразуются в столбцы
        - non-string значения преобразуются в свое представление (repr)

        fix_format:
            нужно ли экранировать символы, ломающие Markdown-таблицы ("|" -> "\\|", "\n" -> "<br>")
        sep_row_full:
            ячейка разделительной строки обязательно должна содержать 3 или более дефиса --- на каждую колонку (спецификаци GitHub Flavored Markdown (GFM)),
            то есть не обязательно должна иметь длину равной имени столбца.
            sep_row_full = True - будет длина как у имени столбца (но не короче ---), иначе ---
    '''
    assert isinstance(data, list) and isinstance(data[-1], dict)

    # сбор имен полей в порядке появления
    fields = []
    seen = set()
    for row in data:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    def format_cell(val: Any) -> str:
        if val is None:
            return ''
        s = repr(val) if isinstance(val, (dict, list)) else str(val)      # optional: json.dumps(val, ensure_ascii=False) вместо repr
        if fix_format:
            return s.replace('|', '\\|').replace('\n', '<br>')
        return s

    fields_row = '| ' + ' | '.join(format_cell(f) for f in fields) + ' |'
    sep_row  = (
        '|' + '|'.join('-' * max(3, len(str(field))) for field in fields) + '|' if sep_row_full
        else '|' + '|'.join('---' for _ in fields) + '|'
    )
    
    body_rows = []
    for row in data:
        cells = [format_cell(row.get(f, '')) for f in fields]
        body_rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join([fields_row, sep_row] + body_rows)