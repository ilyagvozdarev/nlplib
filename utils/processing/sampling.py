def get_windows(words, C):
    """
    a sliding window of length C over a given list of words
    """
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C) : i] + words[(i + 1) : (i + C+ 1)]
        yield context_words, center_word
        i += 1