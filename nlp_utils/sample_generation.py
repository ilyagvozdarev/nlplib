def generate_random_seqs_by_seqs_len_distrib(
    seqs, 
    count, 
    vocab_words
) -> list[list]:
    """
    генерирует заданное количество последовательностей сэмплированием случайных токенов заданного словаря 
    с длинами последовательностей в соответствии с распределением длин последовательностей заданного 
    списка последовательностей

    Parameters
    ----------
    seqs:
        список последовательностей 

    count:
        количество генерируемых последовательностей

    vocab_words:
        частотный словарь

    Returns
    -------
    список последовательностей случайных токенов

 
    Notes
    -----
    формат ner_data:
        {
            "text": текст, 
            "entities": {
                'entity' : тэг сущности, 
                'word' : значение сущности
                // прочие ключи (например start, end)
            }
        }       

    """
    import random

    random_seqs = []
    seq_lengths = list(map(len, seqs))

    for i in range(count):
        len_seq = random.choice(seq_lengths)
        random_seq = random.sample(vocab_words, len_seq)
        random_seqs.append(random_seq)

    return random_seqs




def generate_rus_chars_str(len):
    """
    генерирует строку русских символов заданной длины
    """

    rus_unicode_start = int('0430', 16)
    rus_unicode_end = int('044F', 16)
    s = ''
    for i in range(len):
        s += chr(random.randint(rus_unicode_start, rus_unicode_end + 1))
    return s


