import csv
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache, partial
from typing import Callable, Iterator

import regex
from loguru import logger

from datatrove.utils._import_utils import ASSETS_PATH, check_required_dependencies
from text import TERMINAL_PUNCTUATION

from .rare_langs import *


def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


def simple_span_tokenize(text: str, sents: list[str]) -> Iterator[tuple[int, int]]:
    if len(sents) == 1:
        yield 0, len(text)
        return
    start_index = 0
    for sent in sents:
        start_char = text.index(sent, start_index)
        end_char = start_char + len(sent)
        start_index = end_char
        yield start_char, end_char


# https://github.com/explosion/spaCy/issues/13207
def chunk_text_on_bytes(text: str, max_chunk_size: int = 1_000_000):
    def __utf8len(s: str):
        return len(s.encode("utf-8"))

    factor = len(text) / __utf8len(text) if __utf8len(text) > 0 else 1
    increase_by = int(max(min(max_chunk_size * 0.1, 10), 1))
    initial_size_guess = int(max(max_chunk_size * factor - 10, 1))
    final_list = []
    remaining = text
    while len(remaining):
        part = remaining[:initial_size_guess]
        if __utf8len(part) > max_chunk_size:
            initial_size_guess = max(initial_size_guess - min(max_chunk_size * 0.001, 10), 1)
            continue
        cut_after = initial_size_guess
        while __utf8len(part) < max_chunk_size and part != remaining:
            cut_after = min(len(remaining), cut_after + increase_by)
            part = remaining[:cut_after]

        if __utf8len(part) > max_chunk_size:
            cut_after -= increase_by
        final_list.append(remaining[:cut_after])
        remaining = remaining[cut_after:]

    return final_list


class WordTokenizer(ABC):
    def __init__(self, language: str | None = None):
        self.language = language

    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        pass


class NLTKTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["nltk"])
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk import load

            self._tokenizer = load(f"tokenizers/punkt/{self.language}.pickle")
        return self._tokenizer

    def word_tokenize(self, text) -> list[str]:
        # токенизация на слова (с удалением пустых слов)
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text, language=self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        # токенизация на предложения
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text, language=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        # спаны предложений
        # Notes:
        # - семантика span_tokenize у разных токенизаторов может расходиться:
        #   например у SpaCyTokenizer - спаны не покрывают текст целиком. Span.start_char/end_char считаются по токенам, поэтому пробелы и 
        #   переводы строк между предложениями в спаны не попадают. 
        #   Для "Привет. Мир." получится [(0, 7), (8, 12)] — индекс 7 (пробел) выпадает.
        #   Это отличается от simple_span_tokenize, которым пользуются Thai/Indic/Khmer/Lao/Whitespace: 
        #   там start_index = end_char протягивается по цепочке, и спаны стыкуются встык, включая разделители.
        return list(self.tokenizer.span_tokenize(text))


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, language: str, config=None):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["spacy"])
        if language == "vi":
            check_required_dependencies(f"{language} word tokenizer", ["pyvi"])
        elif language == "zh":
            config = {"nlp": {"tokenizer": {"segmenter": "jieba"}}}
            check_required_dependencies(f"{language} word tokenizer", ["jieba"])
        elif language == "ja":
            # Ensure spaCy uses our locally-registered Japanese tokenizer fix
            # registered in datatrove.utils.japanese_tokenizer as datatrove.ja.JapaneseTokenizer
            # See: https://github.com/explosion/spaCy/issues/13684
            config = {"nlp": {"tokenizer": {"@tokenizers": "datatrove.ja.JapaneseTokenizer"}}}
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import spacy

            # Important to hot-fix the memory leak in Japanese Tokenizer
            from datatrove.utils.japanese_tokenizer import JapaneseTokenizer  # noqa: F401

            if self.config is None:
                self._tokenizer = spacy.blank(self.language)
            else:
                self._tokenizer = spacy.blank(self.language, config=self.config)
            self._tokenizer.add_pipe("sentencizer")
        return self._tokenizer

    def _do_tokenize(self, text: str):
        # japanese has a max byte length
        texts = [text] if self.language != "ja" else chunk_text_on_bytes(text, 40000)
        self.tokenizer.max_length = len(text)
        try:
            return [self.tokenizer(t, disable=["parser", "tagger", "ner"]) for t in texts]
        except Exception as e:
            # this dumb string breaks the tokenizer completely
            if "IS_ALPHA" in text:
                return [self.tokenizer(t.replace("IS_ALPHA", ""), disable=["parser", "tagger", "ner"]) for t in texts]
            else:
                raise e

    def word_tokenize(self, text: str) -> list[str]:
        # Make sure to do all the token processing inside the memory zone, as after that memory address to tokens
        # are not longer valid
        with self.tokenizer.memory_zone():
            self.tokenizer.max_length = len(text) + 10
            tokens = [token.text for tok_chunk in self._do_tokenize(text) for token in tok_chunk]
            return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        with self.tokenizer.memory_zone():
            self.tokenizer.max_length = len(text) + 10
            sents = [sent.text for t in self._do_tokenize(text) for sent in t.sents]
            return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        spans = []
        with self.tokenizer.memory_zone():
            for tok_text in self._do_tokenize(text):
                start = spans[-1][1] if spans else 0
                for sent in tok_text.sents:
                    spans.append((start + sent.start_char, start + sent.end_char))
        return spans


class StanzaTokenizer(WordTokenizer):
    def __init__(self, language: str, **stanza_kwargs):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", ["stanza"])
        self.stanza_kwargs = stanza_kwargs
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import stanza
            from stanza.pipeline.core import DownloadMethod

            self._tokenizer = stanza.Pipeline(
                self.language,
                processors="tokenize",
                download_method=DownloadMethod.REUSE_RESOURCES,
                **self.stanza_kwargs,
            )

        return self._tokenizer

    def word_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        doc = self.tokenizer(text)
        sents = [sentence.text for sentence in doc.sentences]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        doc = self.tokenizer(text)
        return [(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in doc.sentences]


class WhitespaceTokenizer(WordTokenizer):
    """
    This is a fallback tokenizer when no other tokenizer is available.
    """

    def __init__(self):
        super().__init__()
        # should not split on acronyms "(?:\p{{Lu}}\.)"
        self._sent_regex = regex.compile(
            rf"(?:(?:\p{{Lu}}\.)|.)+?[{re.escape(''.join(TERMINAL_PUNCTUATION))}\n]+[\"'”]?", regex.UNICODE
        )

    @property
    @lru_cache(1)
    def _spacy_xx(self):
        # works generally well for white spaces, but does not work to split sentences with a different script
        return SpaCyTokenizer("xx")

    def word_tokenize(self, text) -> list[str]:
        return self._spacy_xx.word_tokenize(text)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = self._sent_regex.findall(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


"""
    The actual tokenizer assignments are saved in src/datatrove/assets/tokenizer_assignments.csv
    If you know a better tokenizer or better proxy language, please submit a PR
"""


@lru_cache(maxsize=1)
def load_tokenizer_assignments() -> dict[str, Callable[[], WordTokenizer]]:
    def tok_factory_wrapper(class_name, arg):
        if class_name == "SpaCyTokenizer":
            tok_class = SpaCyTokenizer
        elif class_name == "StanzaTokenizer":
            tok_class = StanzaTokenizer
        elif class_name == "ThaiTokenizer":
            tok_class = ThaiTokenizer
        elif class_name == "IndicNLPTokenizer":
            tok_class = IndicNLPTokenizer
        elif class_name == "KiwiTokenizer":
            tok_class = KiwiTokenizer
        elif class_name == "KhmerTokenizer":
            tok_class = KhmerTokenizer
        elif class_name == "LaoTokenizer":
            tok_class = LaoTokenizer
        elif class_name == "TibetanTokenizer":
            tok_class = TibetanTokenizer
        elif class_name == "BurmeseTokenizer":
            tok_class = BurmeseTokenizer
        elif class_name == "WhitespaceTokenizer":
            tok_class = WhitespaceTokenizer
        else:
            raise ValueError(f'Invalid tokenizer class "{class_name}"')

        if arg:
            return tok_class(arg)
        return tok_class()

    word_tokenizer_factories = {}
    with open(os.path.join(ASSETS_PATH, "tokenizer_assignment.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_3, code_1, script, tok_class_name, tok_code, default_script, default_code_1 = (
                row["code_3"],
                row["code_1"],
                row["script"],
                row["type"],
                row["tok_code"],
                row["default_script"],
                row["default_code_1"],
            )

            if not tok_class_name:
                continue

            tok_factory = partial(tok_factory_wrapper, tok_class_name, tok_code)

            code_3_script = f"{code_3}_{script}"
            if code_3_script not in word_tokenizer_factories:
                word_tokenizer_factories[code_3_script] = tok_factory
                if default_script:
                    word_tokenizer_factories[code_3] = tok_factory
            code_1_script = f"{code_1}_{script}"
            if code_1 and default_code_1 and code_1_script not in word_tokenizer_factories:
                word_tokenizer_factories[code_1_script] = tok_factory
                if default_script:
                    word_tokenizer_factories[code_1] = tok_factory

    return word_tokenizer_factories


@lru_cache(maxsize=None)
def load_word_tokenizer(language_or_tok: str | WordTokenizer) -> WordTokenizer:
    if isinstance(language_or_tok, WordTokenizer):
        # for custom tokenizers
        return language_or_tok
    word_tokenizer_factories = load_tokenizer_assignments()
    if language_or_tok not in word_tokenizer_factories:
        raise ValueError(
            f"Language '{language_or_tok}' doesn't have a tokenizer assigned. Pass in a "
            f"WordTokenizer directly or update tokenizer_assignment.csv"
        )
    return word_tokenizer_factories[language_or_tok]()
