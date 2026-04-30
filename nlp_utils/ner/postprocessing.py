from typing import Callable
from ..processing.preprocessing import pymorph_normalize


def process_ner_output(
    ner_data: dict, 
    *,
    normalizer: Callable = pymorph_normalize(),
    remove_punct: bool = True
):
    """
    Преобразует dict-like информацию о сущностях текста в DataFrame и делает постобработку

    Parameters
    ----------
    ner_data:
        словарь с текстом и его сущностями

    normalizer:
        функция нормализующая слово

    remove_punct:
        нужно ли удалять знаки пунктуации из строки-значения сущности

    Returns
    -------
    DataFrame:
        статистика по сущностям 

        text: текст
        entity: тэг сущности
        word: значение сущности
        word_normal: значение сущности в нормальной форме
        word_clear_punct: значение сущности без знаков пунктуации
        // прочие ключи

 
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
    import pandas as pd
    import string

    df_ners = pd.DataFrame(ner_data).apply(lambda row: {'text' : row['text'], **row['entities']}, 
                                            axis = 1, 
                                            result_type = 'expand')
    if remove_punct:
        df_ners['word_clear_punct'] = df_ners['word'].apply(
            lambda row : row.translate(str.maketrans('', '', string.punctuation))
        )

    df_ners['word_normal'] = df_ners['word_clear_punct'].apply(
        lambda row : normalizer(row)
    )
    return df_ners


def entities_by_ner(
    ner_data,
    col_nertag,
    col_entity
):
    """
    формирует словарь значений сущности из датафрейма из заданных столбцов (тэг сущности, значение сущности)

    Parameters
    ----------
    ner_data:
        DataFrame с значениями сущностей
    
    col_nertag:
        столбец с тэгом сущности

    col_entity:
        столбец со значением сущности

    Returns
    -------
    dict:
        словарь значений сущности
        key: тэг сущности
        value: значения сущностей данного тэга  

    """    

    return ner_data.groupby(by = col_nertag).apply(lambda df : df[col_entity].values).to_dict()



def entities_intersects(
    entities1: dict,
    entities2: dict
):
    """
    формирует статистику по пересечению тэгов и значений сущностей между 
    двумя словарями сущностей

    Parameters
    ----------
    entities1:
        словарь значений сущностей
    
    entities2:
        словарь значений сущностей

    Returns
    -------
    dict:
        словарь со статистикой по пересечению тэгов и значений сущностей
        tags1_not_in_tags2: тэги первого словаря, которых нет во втором
        tags2_not_in_tags1: тэги второго словаря, которых нет в первом
        common_tags: тэги которые есть в обеих словарях
        entities_intersects:
            *тэг1*:
                common_entities: словарь количества значений сущностей общих тэгов которые есть в обеих словарях
                    *сущность1*: количество встреч
                    *сущность2*: количество встреч
                    ...
                entities_1_not_in_2: словарь количества значений сущностей общих тэгов которые есть в пером словаре, но нет во втором
                entities_2_not_in_1: словарь количества значений сущностей общих тэгов которые есть во втором словаре, но нет в пером
            *тэг2*:
            ...

    Notes
    -----
    формат entities1, entities2:
        key: тэг сущности
        value: значения сущностей данного тэга
    """    

    from collections import defaultdict
    intersects = defaultdict(dict)

    intersects['tags1_not_in_tags2'] = set(entities1.keys()) - set(entities2.keys())
    intersects['tags2_not_in_tags1'] = set(entities2.keys()) - set(entities1.keys())
    intersects['common_tags'] = set(entities1.keys()) & set(entities2.keys())

    for tag in intersects['common_tags']:
        
        from collections import Counter
        entities1_co = Counter(entities1[tag])
        entities2_co = Counter(entities2[tag])

        entities_1_not_in_2 = entities1_co - entities2_co
        entities_2_not_in_1 = entities2_co - entities1_co
        common_entities = entities1_co & entities2_co

        intersects['entities_intersects'][tag] = {
                'common_entities' : dict(common_entities),
                'entities_1_not_in_2' : dict(entities_1_not_in_2),
                'entities_2_not_in_1' : dict(entities_2_not_in_1)
        }

    return intersects



