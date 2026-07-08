import json, os, yaml
from typing import Callable, Any
from pathlib import Path


def read(file, reader):
    with open(file, encoding="utf-8") as f:
        try:
            return reader(f)
        except Exception as e:
            print(f'file = {file}')
            raise e    

def read_json(file):
    return read(file, json.load)

def read_yaml(file):
    return read(file, yaml.safe_load)

def read_jsonl(file):
    with open(file, encoding="utf-8") as r:
        return [json.loads(line) for line in r]

def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(obj, filename, overwrite=True, indent=2):
    # если overwrite=False то данные не будут перезаписываться если файл уже существует
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if overwrite or not overwrite and not os.path.isfile(filename):
        with open(filename, 'w') as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

def write_yaml(obj, filename, overwrite=True):
    # если overwrite=False то данные не будут перезаписываться если файл уже существует
    import yaml
    class IndentDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentDumper, self).increase_indent(flow, False)
    if overwrite or not overwrite and not os.path.isfile(filename):
        with open(filename, 'w') as f:
            yaml.dump(obj, f, allow_unicode=True, Dumper=IndentDumper, width=float("inf"))

def write_txt(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def read_config(file):
    ext = os.path.splitext(file)[-1].lower()
    exts = ['.json', '.yaml', '.yml']
    assert ext in exts, f'config ext not in {exts}'
    return (read_json if ext == '.json' else read_yaml)(file)


def get_files_from_dir(
  dir, 
  exclude_dir_names = [],
  exts=['txt', 'json', 'yaml']  
):
    # exts = None или [] - возращает все файлы

    exts_found = set()

    if exts is None:
        exts = []
    files_res = []
    walk = os.walk(dir, topdown=True, onerror=None, followlinks=False)

    for root, _, files in walk:
        intermediate_dirs = root.split(os.sep)
        need_exclude_dir = any(set(intermediate_dirs) & set(exclude_dir_names))
        if not need_exclude_dir: 
            files_ = []
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in exts or len(exts) == 0:
                    files_.append(root + os.sep + file)
                    exts_found.add(ext)
            files_res.extend(files_)
    return files_res, exts_found


def read_sources(
    files, 
    reader: Callable[..., Any] | dict[str, Callable[..., Any]],
    add_filename = True
):
    # reader: reader или маппинг расширение -> reader
    data = []
    reader_ = reader
    for file in files:
        _, filename = os.path.split(file)
        if isinstance(reader, dict):
            reader_ = reader[os.path.splitext(filename)[1][1:]]
        data_file = reader_(file)
        if add_filename:
            data_file = (filename, data_file)
        data.append(data_file)
    return data


def print_files_tree(dir_path: Path, indent: int = 0):
    entries = sorted(dir_path.iterdir(), key=lambda e: (e.is_file(), e.name))
    for entry in entries:
        print("   " * indent + "|- " + entry.name)
        if entry.is_dir():
            print_files_tree(entry, indent + 1)

