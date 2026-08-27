import re
from loguru import logger

from word_tokenizers import (
    WordTokenizer, strip_strings, simple_span_tokenize, WhitespaceTokenizer
)
from datatrove.utils._import_utils import check_required_dependencies


class ThaiTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("th word tokenizer", ["pythainlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize as th_word_tokenize

        tokens = th_word_tokenize(text, keep_whitespace=False, engine="newmm-safe")
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from pythainlp.tokenize import sent_tokenize as th_sent_tokenize

        sents = th_sent_tokenize(text)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class IndicNLPTokenizer(WordTokenizer):
    def __init__(self, language: str):
        super().__init__(language)
        check_required_dependencies(f"{language} word tokenizer", [("indicnlp", "indic-nlp-library")])

    def word_tokenize(self, text) -> list[str]:
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indicnlp_trivial_tokenize

        tokens = indicnlp_trivial_tokenize(text, self.language)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from indicnlp.tokenize.sentence_tokenize import sentence_split

        sents = sentence_split(text, lang=self.language)
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class KiwiTokenizer(WordTokenizer):
    def __init__(self, model_type="sbg"):
        super().__init__()
        check_required_dependencies("ko word tokenizer", ["kiwipiepy"])
        self.model_type = model_type
        self._tokenizer = None
        self._preprocess_regex = re.compile("[0-9,]{20,}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from kiwipiepy import Kiwi

            self._tokenizer = Kiwi(model_type=self.model_type)
        return self._tokenizer

    def preprocess(self, text):
        # seems to have issue with very large numbers
        return self._preprocess_regex.sub("", text)

    def word_tokenize(self, text: str) -> list[str]:
        tokens = [text[token.start : token.end] for token in self.tokenizer.tokenize(self.preprocess(text))]
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        sents = [sent.text for sent in self.tokenizer.split_into_sents(self.preprocess(text))]
        return strip_strings(sents)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(sent.start, sent.end) for sent in self.tokenizer.split_into_sents(self.preprocess(text))]


class KhmerTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("khmer word tokenizer", [("khmernltk", "khmer-nltk")])

    def word_tokenize(self, text: str) -> list[str]:
        from khmernltk import word_tokenize

        tokens = word_tokenize(text, return_tokens=True)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from khmernltk import sentence_tokenize

        return strip_strings(sentence_tokenize(text))

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class LaoTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("laos word tokenizer", ["laonlp"])

    def word_tokenize(self, text: str) -> list[str]:
        from laonlp.tokenize import word_tokenize

        tokens = word_tokenize(text)
        return strip_strings(tokens)

    def sent_tokenize(self, text: str) -> list[str]:
        from laonlp.tokenize import sent_tokenize

        return strip_strings(sent_tokenize(text))

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))


class TibetanTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("tibetan word tokenizer", ["botok"])
        self._wt = None
        self._whitespace_regex = re.compile(r"\s+")

    @property
    def wt(self):
        if self._wt is None:
            from botok import WordTokenizer

            self._wt = WordTokenizer()
        return self._wt

    def _try_tokenize(self, text: str) -> list[str]:
        try:
            return self.wt.tokenize(text, split_affixes=False)
        except Exception as e:
            logger.warning(f"Failed to tokenize with botok: {e}. Trying without spaces...")
            return self.wt.tokenize(self._whitespace_regex.sub("", text), split_affixes=False)

    def word_tokenize(self, text: str) -> list[str]:
        return strip_strings([tok.text for tok in self._try_tokenize(text)])

    def sent_tokenize(self, text: str) -> list[str]:
        from botok.tokenizers.sentencetokenizer import sentence_tokenizer

        tokens = self._try_tokenize(text)
        sents = sentence_tokenizer(tokens)
        out = ["".join([word.text for word in s["tokens"]]) for s in sents]
        return strip_strings(out)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        from botok.tokenizers.sentencetokenizer import get_sentence_indices

        tokens = self._try_tokenize(text)
        idxs = get_sentence_indices(tokens)
        return [(sentence["start"], sentence["end"] + 1) for sentence in idxs]


class BurmeseTokenizer(WhitespaceTokenizer):
    def __init__(self):
        super().__init__()
        check_required_dependencies("burmese word tokenizer", [("pyidaungsu", "pyidaungsu-numpy2")])
        self._wt = None

    def word_tokenize(self, text: str) -> list[str]:
        import pyidaungsu as pds

        tokens = pds.tokenize(text, form="word")
        return strip_strings(tokens)