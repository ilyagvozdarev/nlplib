"""Function for the Levenshtein algorithm.

Note: This Python implementation is very inefficient. Please use this C++
implementation instead: https://github.com/wq2012/word_levenshtein
"""
import numpy as np
from enum import Enum
import json, os
from typing import Callable



def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]

def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def export_to_json(obj, filename, overwrite=True):
    # если overwrite=False то данные не будут перезаписываться если файл уже существует
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if overwrite or not overwrite and not os.path.isfile(filename):
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

def export_to_yaml(obj, filename, overwrite=True):
    # если overwrite=False то данные не будут перезаписываться если файл уже существует
    import yaml
    class IndentDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentDumper, self).increase_indent(flow, False)
    if overwrite or not overwrite and not os.path.isfile(filename):
        with open(filename, 'w') as f:
            yaml.dump(obj, f, allow_unicode=True, Dumper=IndentDumper, width=float("inf"))


def load_json(filename):
    with open(filename) as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f'filename = {filename}')
            raise e


def get_files_from_dir(
  dir, 
  exclude_dir_names = [],
  exts=['txt', 'json', 'yaml']  
):
    from nlp_utils.regexp import split_file_ext

    files_res = []
    walk = os.walk(dir, topdown=True, onerror=None, followlinks=False)

    for root, _, files in walk:
        intermediate_dirs = root.split(os.sep)
        need_exclude_dir = any(set(intermediate_dirs) & set(exclude_dir_names))
        if not need_exclude_dir: 
            files_res.extend([
                root + os.sep + file 
                for file in files 
                if split_file_ext(file)[1] in exts
            ])
    return files_res


def load_data_from_files(
    files_list, 
    extractor: Callable,
    added_filename_key = None
):
    data = []
    for file in files_list:
        file_dir, filename = os.path.split(file)
        examples = extractor(file)
        for example in examples:
            data_d = {**example}
            if added_filename_key:
                data_d.update({added_filename_key: filename})
            data.append(data_d)
    return data

