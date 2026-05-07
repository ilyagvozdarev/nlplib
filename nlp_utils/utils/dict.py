from enum import Enum
import json, os
from typing import Callable, Iterable
from collections.abc import MutableMapping



def rename_dict_keys_deep(data, key_mapping):
    """
        Рекурсивно переименовывает ключи в словаре и во всех вложенных словарях.

        :param data: исходный словарь или список
        :param key_mapping: словарь сопоставления новых ключей в старые:
            {
                'new_key1': ['old_key1', 'old_key2'],
                'new_key2': ['old_key3'],
            }
        :return: новый словарь/список с переименованными ключами
    """

    def new_key(key, key_mapping):
        for key_new, keys_old in key_mapping.items():
            if key in keys_old:
                return key_new
        return key


    if isinstance(data, dict):
        return {
            new_key(key, key_mapping): rename_dict_keys_deep(value, key_mapping)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [rename_dict_keys_deep(item, key_mapping) for item in data]
    else:
        return data


def flatten_dict(dictionary, separator='__'):
    return {separator.join(map(str, keys)): val for keys, val in dict_to_flatten_keys(dictionary)}



def dict_to_flatten_keys(dictionary, parent_keys=None):
    # dict -> список из (список ключей до non-dict значения, non-dict значение)
    if parent_keys is None:
        parent_keys = []
    items = []
    for key, value in dictionary.items():
        if isinstance(value, MutableMapping):
            items.extend(dict_to_flatten_keys(value, parent_keys + [key]))
        else:
            items.append((parent_keys + [key], value))
    return items




def is_keys_values_in_dict(
        dict: dict, 
        filter_dict: dict
):
    '''
        проверяет содержатся ли ключи-значения filter_dict в ключах-значениях dict
    '''
    return not any([
        filter_k not in dict or filter_dict[filter_k] != dict[filter_k] 
        for filter_k in filter_dict
    ])


def get_dicts_by_filter_dicts(
        dicts: list[dict], 
        filter_dicts: list[dict]
):
    '''
        принимает список словарей и возвращает словари у которых
        значения по указанным ключам содержатся в списке ключей-значений
        заданного словаря
    
        dicts - фильтруемый список словарей
        filter_dicts - список словарей для фильтрации
    '''
    res = []
    for d in dicts:
        for filter_d in filter_dicts:
            if is_keys_values_in_dict(d, filter_d):
                res.append(d)
                break
    return res


def count_nested(d, count_to='elem'):
    # рекурсивный подсчет количества записей в словаре
    # count_to='elem': если значение iterable то считает количество элементов
    count = 0
    for v in d.values():
        if isinstance(v, dict):
            count += count_nested(v)
        elif isinstance(v, Iterable) and count_to == 'elem':
            count += len(v) 
        else:
            count += 1
    return count