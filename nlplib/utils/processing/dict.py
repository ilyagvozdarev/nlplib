from typing import Iterable
from collections.abc import MutableMapping


def rename_dict_keys_deep(data, key_mapping):
    """
    Recursively renames keys in a dictionary and in all nested dictionaries.

    Parameters
    ----------
    data: 
        the source dictionary or list
    key_mapping: 
        a mapping of new keys to old keys:
        {
            'new_key1': ['old_key1', 'old_key2'],
            'new_key2': ['old_key3'],
        }
    
    Returns
    -------
    a new dictionary/list with renamed keys
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


def dict_to_flatten_keys(d, parent_keys=None, nested_key=None):
    """
    a list of (list of keys leading to a non-dict value, non-dict value)
    """
    if parent_keys is None:
        parent_keys = []
    items = []
    for key, value in d.items():
        if isinstance(value, MutableMapping):
            items.extend(dict_to_flatten_keys(value, parent_keys + [key]))
        else:
            items.append((parent_keys + [key], value))
    return items


def is_keys_values_in_dict(dict: dict, filter_dict: dict):
    """
    checks whether the keys and values ​​of filter_dict are contained in the keys and values ​​of dict
    """
    return not any([
        filter_k not in dict or filter_dict[filter_k] != dict[filter_k] 
        for filter_k in filter_dict
    ])


def count_nested(d, count_to='elem'):
    """
    Recursively counts the number of entries in a dictionary.
    count_to='elem': if the value is iterable, counts the number of
    elements in it.
    """
    count = 0
    for v in d.values():
        if isinstance(v, dict):
            count += count_nested(v)
        elif isinstance(v, Iterable) and count_to == 'elem':
            count += len(v) 
        else:
            count += 1
    return count


def apply_nested(d, func, nested_key='nested', parents=None):
    """
    Applies func to the dictionary and then recursively to the value
    under the 'nested' key.

    A list is processed element by element, so the 'nested' value may be
    either a dictionary or a list of dictionaries.

    Parameters
    ----------
    d:
        the source dictionary or list
    func:
        a callable dict, parents -> dict
    nested_key:
        the key whose value is processed recursively
    parents:
        list of parent dicts

    Returns
    -------
    a new dictionary/list; neither the source data nor func's result is modified
    """
    parents = parents or []
    if isinstance(d, list):
        return [apply_nested(item, func, nested_key, parents) for item in d]
    if not isinstance(d, dict):
        return d
    result = func(d, parents)
    if nested_key in result:
        result = {**result, nested_key: apply_nested(result[nested_key], func, nested_key, parents + [result])}
    return result
