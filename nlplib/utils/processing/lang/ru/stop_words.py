from enum import StrEnum
from functools import lru_cache


EN_TEST: frozenset[str] = frozenset({"the", "a", "of"})
DOMAIN_TEST: frozenset[str] = frozenset({"артикул", "sku"})


class StopWordSet(StrEnum):
    RU_SPACY = "ru_spacy"
    RU_NLTK = "ru_nltk"
    EN_SKLEARN = "en_sklearn"
    EN_TEST = "en"
    DOMAIN_TEST = "domain"


@lru_cache(maxsize=None)
def _load(name: StopWordSet) -> frozenset[str]:
    match name:
        case StopWordSet.EN_TEST:
            return EN_TEST
        case StopWordSet.DOMAIN_TEST:
            return DOMAIN_TEST
        case StopWordSet.RU_SPACY:
            from ru_spacy import RU_SPACY
            return frozenset(RU_SPACY)
        case StopWordSet.RU_NLTK:
            from ru_nltk import RU_NLTK
            return frozenset(RU_NLTK)
        case StopWordSet.EN_SKLEARN:
            from sklearn.feature_extraction import _stop_words
            return _stop_words.ENGLISH_STOP_WORDS
    raise NotImplementedError(f"не задан источник для {name!r}")


def get_stop_words(*names: StopWordSet | str) -> frozenset[str]:
    members = [StopWordSet(n) for n in names]
    return frozenset().union(*(_load(m) for m in members))