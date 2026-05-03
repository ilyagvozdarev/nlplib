from abc import ABC, abstractmethod


def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]


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


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, language: str, config=None):
        super().__init__(language)
        self.config = config
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import spacy
            config_kwargs = {"config": self.config} if self.config is not None else {}
            self._tokenizer = spacy.blank(self.language, **config_kwargs)
            self._tokenizer.add_pipe("sentencizer")
        return self._tokenizer

    def _do_tokenize(self, text: str):
        texts = [text]
        self.tokenizer.max_length = len(text)
        try:
            return [self.tokenizer(t, disable=["parser", "tagger", "ner"]) for t in texts]
        except Exception as e:
            # this dumb string breaks the tokenizer completely
            if "IS_ALPHA" in text:
                return [self.tokenizer(t.replace("IS_ALPHA", ""), disable=["parser", "tagger", "ner"]) for t in texts]
            else:
                raise e

    #todo: refactor repeats

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