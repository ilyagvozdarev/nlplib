import re, pymorphy3
from num2words import num2words


morph = pymorphy3.MorphAnalyzer()


QUANTITIVE_SUFFIXES = {
    'gent': ['го', 'ого', 'ного', 'ух', 'х', 'ех', 'ёх', 'и', 'ти', 'ми'],        
    'datv': ['у', 'му', 'м', 'ум', 'ем', 'ём', 'и', 'ти', 'ми', 'ну'],
    'accs': ['го', 'ого', 'ух', 'х', 'ех'],
    'ablt': ['м', 'им', 'ним', 'я', 'мя', 'ью', 'ю'],
    'loct': ['м', 'ом', 'х', 'ух', 'ех', 'и', 'ти', 'ми']
}

ORDINAL_SUFFIXES = {
    'masc': {
        'nomn': ['ой', 'ый', 'ий', 'й', 'ый'],
        'gent': ['ого', 'го', 'его'],
        'datv': ['ому', 'у', 'му', 'ему'],
        'ablt': ['м', 'ым', 'им'],
        'loct': ['ом', 'ем']
    },
    'femn': {
        'nomn': ['я', 'ая'],
        'gent': ['ой', 'й', 'ей'],      # datv/ablt/loct для жен.рода имеют такие же формы слов как для gent
        'accs': ['ю', 'ью', 'ую'],      # (Пример: третью)
    },
    'neut': {
        'nomn': ['е', 'ое', 'ье']
    }
}

# QS совпадающие с OS:
# !!!НЕ задавать в text для количественных чтобы не было неоднозначности
# Примеры:
#   5-м -> пятом (пути), пятым (башмаком)
#   31-го -> тридцати одного, тридцать первого
#   3-я -> третья, тремя
#   -ью -> пятью, третью
QS_in_OS = ['го', 'ого', 'у', 'му', 'м', 'ем', 'ю', 'ью', 'я', 'им', 'ом']


QS = [s for suffixes in QUANTITIVE_SUFFIXES.values() for s in suffixes]
OS = [s for cases in ORDINAL_SUFFIXES.values() for suffixes in cases.values() for s in suffixes]
SUFFIXES = set(OS) | set(QS)



# Patterns

VAL_TAG_P = r'(\[[^\[\]]+?\])'      # значение в [...] в котором отсутствуют "[]"
VAL_AND_TAG_P = rf'(?<!]){VAL_TAG_P}{VAL_TAG_P}+'
TAGS = ['tag1', 'tag2']
TAG_P = rf'(\[(?:\d+-)?(?:{"|".join(TAGS)})\])'

QS_P = '(' + '|'.join(QS) + ')'
QS_in_OS_P = '(' + '|'.join(QS_in_OS) + ')'
QS_in_OS_num_P = rf'(?:\b|^)\d+[-–]{QS_in_OS_P}(?:\b|$)'         # числительные с суффиксом который подходит для количественной и порядковой формы
num_and_suf_P = r'(?:\b|^)\d+[-–]([А-Яа-яЁё]{0,3})(?:\b|$)'      # числительные с суффиксом

TIME_P = r'(\d+)\s*(ч\.|час\.?|ч)((\s*)(\d+)\s*(м\.|мин\.?|м))?(?=\b|\s|$)'
TIME_P_2 = r'(?<=(?:\b|^))(с\s+)?(\d{2}):(\d{2})(?=(?:\b|$))'
TIME_MIN_P = r'(\d+)\s*(м\.|мин\.?|м)(?=\b|\s|$)'
TIME_WORDS_P = r'(?<=(?:\b|^))(с\s+)?(\d+)\s+(час(?:а|ов)?)\s+(\d+)\s+(минут(?:а|ы)?)(?=(?:\b|$))'
DATE_P = rf'(\d{1,2}\.\d{1,2}\.\d{2,4})г\.'


def remove_entities(text):
    return re.sub(VAL_TAG_P + TAG_P, r'\2', text, flags=re.IGNORECASE)


def agree_hours_mins_words_with_number(match):
    time_and_noun = [[int(match.group(1)), 'час']]
    if match.group(3):
        time_and_noun.append([int(match.group(5)), 'минута'])
    agreed = ''
    for time, time_noun in time_and_noun:
        parsed = morph.parse(time_noun)[0]
        time_noun = parsed.make_agree_with_number(time).word

        agreed += f'{time} {time_noun} '
    return agreed.strip()


def time_to_word_format(match):
    grammemes = []
    # print(match.groups())
    if match.group(1):
        case = 'gent'
        grammemes.append([case, 'plur'] if match.group(2)[-1] != '1' else [case])
        grammemes.append([case, 'plur'] if match.group(3)[-1] != '1' else [case])

    hours, mins = int(match.group(2)), int(match.group(3))
    return (match.group(1) or '') + time_to_word_format_(hours, mins, grammemes)


def time_to_word_format_(hours, mins, grammemes=None):
    agreed = ''
    for time_noun, time in [('час', hours), ('минута', mins)]:
        parsed = morph.parse(time_noun)[0]
        if grammemes:
            parsed = parsed.inflect(set(grammemes[0 if time_noun == 'час' else 1]))
            # print(parsed)
        time_noun = parsed.make_agree_with_number(time).word
        agreed += f'{time} {time_noun} '
    return agreed.strip()


def get_suf_case(num_s, case):
    assert int(num_s)
    suffixes = {
        'loct': ['-ом', '-ем'],
        'gent': ['-ого', '-его'],
        'datv': ['-ому', '-ему'],
        'ablt': ['-ым', '-им']
    }
    if num_s[-1] == '3':
        if len(num_s) > 1 and num_s[-2] == '1':
            return suffixes[case][0]
        return suffixes[case][1]
    return suffixes[case][0]



# to ASR format

def get_grammems_by_suffix_OS(suffix):
    for gen, cases in ORDINAL_SUFFIXES.items():
        for case, suffixes in cases.items():
            if suffix in suffixes:
                return {'gen': gen, 'case': case}
    return None


def get_grammems_by_suffix_QS(suffix):
    for case, suffixes in QUANTITIVE_SUFFIXES.items():
        if suffix in suffixes:
            return {'case': case}
    return None


def select_numr_parsing(word):
    # для числительного прописью выбирает разбор (NUMR) так как может быть 
    # выбран NOUN как самое вероятное, например для 'сто' или при отсутствии 
    # выбирается (ADJF) (например для 'один')
    parsed = morph.parse(word)
    parse_res = [p for p in parsed if p.tag.POS == 'NUMR']
    
    if not parse_res:
        parse_res = [p for p in parsed if p.tag.POS == 'ADJF']

    if not parse_res:
        print('\n!!! Не найден разбор (NUMR) для числительного прописью, выбирается самый вероятный разбор')
        print('слово = ' + word + '\nразборы: ')
        for p in parsed:
            print(p)
        parse_res = parsed[0]
        return parse_res
        
    return parse_res[0]


def num2words_case(num, case='gent'):
    """
    переводит число в пропись в заданном падеже
    """
    num_words = []
    for word in num2words(num, lang='ru').split():
        numr_parse = select_numr_parsing(word)
        num_words.append(numr_parse.inflect({case}).word)
    return ' '.join(num_words) 


def process_match_num2words(match):
    """
    переводит числа (втч с суффиксами), записанные арабскими цифрами 
    (e.g. "4090", "5-го пути") в написание прописью.
    """
    num_str = match.group(1)
    suffix = match.group(2)
    num = int(num_str)

    if not suffix:
        # Простые количественные числительные
        return num2words(num, lang='ru')

    grammemes = get_grammems_by_suffix_OS(suffix)

    if grammemes:
        # суффиксы которые подходят для quantitive и ordinal считаем ordinal 
        # преобразуем в порядковое определенного рода 
        gender = 'f' if 'femn' == grammemes['gen'] else ('m' if 'masc' == grammemes['gen'] else 'n')
        # print(grammemes)
        # print(gender)
        ordinal = num2words(
            num, 
            lang='ru', 
            to='ordinal', 
            gender=gender
        ).split()
        last_num_word = ordinal[-1]
        # print(last_num_word)
        
        # Склоняем полученное числительное:
        # нужно склонять последнее слово: 326 -> триста двадцать шесть -> триста двадцать шестой
        numr_parse = select_numr_parsing(last_num_word)
        last_num_word = numr_parse.inflect({grammemes['case'], grammemes['gen']}).word
        ordinal[-1] = last_num_word
            
        return ' '.join(ordinal)

    # Обработка количественных числительных:
    # нужно склонять каждое слово: 326 -> триста двадцать шесть -> трехсот двадцати шести
    grammemes = get_grammems_by_suffix_QS(suffix)
    # print(grammemes)
    if grammemes:
        words = []
        for word in num2words(num, lang='ru').split():
            numr_parse = select_numr_parsing(word)
            words.append(numr_parse.inflect({grammemes['case']}).word)
        return ' '.join(words)


def replace_num2words(text):
    # шаблон для чисел (втч с суффиксами)
    pattern = rf'(?:\b|^)(\d+)(?:[-–]({"|".join(SUFFIXES)}))?(?:\b|$)'
    return re.sub(pattern, process_match_num2words, text, flags=re.IGNORECASE)


def separate_numbers_and_letters(text):
    # Разделение слитных чисел и букв (например, "33В", "м23", "М174", "Ч123Д")
    # суффикс у букв ставится к последней цифре: "30 В-й" -> 30-й В
    letter_p = r'[А-Яа-яЁёA-Za-z]'
    pattern = re.compile(rf'(\d+{letter_p}+|{letter_p}+\d+)')
    
    def replacer(match):
        res = match.group()
        res = re.sub(rf'(\d)({letter_p})', r'\1 \2', res)
        res = re.sub(rf'({letter_p})(\d)', r'\1 \2', res)
        return res

    while re.search(pattern, text):
        text = pattern.sub(replacer, text)

    parts = text.split()
    parts_res = parts.copy()
    for i, part in enumerate(parts):
        if match := re.search(rf'(^.*)[-–]({"|".join(OS)})$', part):
            # print('match = ' + text)
            if not match.group(1).isdigit():
                assert parts[i-1].isdigit(), (parts[i-1], match)
                parts_res[i-1] += '-' + match.group(2)
                parts_res[i] = match.group(1)
    res = ' '.join(parts_res)
    return res


def num_min_to_words(minutes):
    # перевод минут в пропись (0-60 -> ...)
    if 10 <= minutes <= 20:
        # Для чисел 10-20 род всегда мужской
        number_words = num2words(minutes, lang='ru')
    else:
        last_digit = minutes % 10
        if last_digit == 1:
            # одна, двадцать одна, тридцать одна и т.д.
            number_words = num2words(minutes, lang='ru', gender='f')
        elif last_digit == 2:
            # Для двойки нужно заменить "два" на "две" в женском роде
            base_words = num2words(minutes, lang='ru')
            number_words = base_words.replace("два", "две", 1)
        else:
            number_words = num2words(minutes, lang='ru')
    return number_words


def time_to_words(match):
    num_words = (
        num2words(match.group(2), lang='ru') 
        + ' ' + ('ноль ' if match.group(3)[0] == '0' else '') 
        + num2words(match.group(3), lang='ru')
    )
    if match.group(1):
        # родительный падеж
        words = []
        for word in num_words.split():
            numr_parse = select_numr_parsing(word)
            words.append(numr_parse.inflect({'gent'}).word)
        return (match.group(1) or '') + ' '.join(words)
    return (match.group(1) or '') + num_words

    
def time_words_to_words(match):
    hours_words = num2words(match.group(2), lang='ru', gender='m')
    mins_words = num_min_to_words(int(match.group(4)))
    if match.group(1):
        # родительный падеж
        words_gent_hm = []
        for words in [hours_words, mins_words]:
            words_gent = []
            for word in words.split():
                numr_parse = select_numr_parsing(word)
                words_gent.append(numr_parse.inflect({'gent'}).word)
            words_gent_hm.append(' '.join(words_gent))
        hours_words, mins_words = words_gent_hm
    return (match.group(1) or '') + hours_words + ' ' + match.group(3) + ' ' + mins_words + ' ' + match.group(5)


def reg_to_asr(text):
    """
    - Разделение слитных чисел и букв (например, "33В", "м23", "М174", "Ч123Д"),
      суффикс у букв ставится к последней цифре: "30 В-й" -> 30-й В
    - № 123 -> номер 123
    - ё -> е
    - 123 км/ч -> 123 километра в час
    - км -> километр
    - 12:34 -> двенадцать тридцать четыре | 12:04 -> двенадцать ноль четыре | с 12:34 -> с двенадцати тридцати четырех
    - 23 часа 31 минута -> двадцать три часа тридцать одна минута
    - с 23 часов 31 минуты -> с двадцати трех часов тридцать одной минуты
    """
    text = separate_numbers_and_letters(text)

    # № 123 -> номер 123
    # text = re.sub(r'№((\s)|(\S)|$)', lambda m: 'номер ' if m.group(3) else 'номер', text)
    text = re.sub(r'№', 'номер', text)
    text = re.sub('ё', 'е',  text)

    # 123 км/ч -> 123 километра в час
    km_parsed = morph.parse('километр')[0]

    def _km_to_words(m):
        number = int(m.group(1))
        word = km_parsed.make_agree_with_number(number).word
        return f'{number} {word} в час'

    text = re.sub(r'(\d+)\s*(км/час|км/ч)\b', _km_to_words, text)

    if 'км/ч' in text:
        print(text)

    # км
    text = re.sub(r'(?:\b|^)км(?:\b|$)', 'километр', text)

    # 15:43 -> пятнадцать сорок три
    # 05:43 -> пять сорок три
    # с 15:43 -> с пятнадцати сорока трех
    text = re.sub(TIME_P_2, time_to_words, text, flags=re.IGNORECASE)

    # 23 часа 31 минута -> двадцать три часа тридцать одна минута
    # с 23 часов 31 минуты -> с двадцати трех часов тридцать одной минуты
    text = re.sub(TIME_WORDS_P, time_words_to_words, text, flags=re.IGNORECASE)

    return text


if __name__ == "__main__":
    import re, pymorphy3
    morph = pymorphy3.MorphAnalyzer()

    test_texts = [
        'с 23:41',
        '23:41',
        'с 11:43',
        '11:43',
        '01:43',
        'с 01:43'
    ]

    for text in test_texts:
        print(re.sub(TIME_P_2, time_to_word_format, text))

    test_texts = [
        '22 час. 15 м.',
        '1 ч. 32 мин.',
        'Адлер, в 10час.20мин на 43 пути',
        '6 ч 7 мин',
        '6 часов 7 минут',
        '6 ч.',
        '7 мин',
        '23 часа 23 мин'
    ]

    for text in test_texts:
        print(re.sub(TIME_P, agree_hours_mins_words_with_number, text))

    test_texts = [
        '7 мин',
        '23 часа 23 мин',
        '23 м',
        '1 мин.'
    ]

    for text in test_texts:
        text = re.sub(
            TIME_MIN_P, 
            lambda m: f'{int(m.group(1))} {morph.parse("минута")[0].make_agree_with_number(int(m.group(1))).word}', 
            text
        )
        print(text)


    test_texts = [
        "Машинист № 9623-го Шадрин, слушаю.",
        "Ч123Д  Ч1  1Ч  30В-й",
        "20–го"
    ]

    for text in test_texts:
        print(separate_numbers_and_letters(text))
        print(replace_num2words(text))
        print()

    test_texts = [
        '326-ая',
        '326-ого',
        '326-ти',
        'из 121 вагона',
        '121 вагон',
        'из 26 вагонов',
        'с 121 вагоном',
        'с 26 вагонами',
        'с 21 вагоном',
        '1 тормозным башмаком',
        '2 тормозными башмаками',
        '3 тормозными башмаками'
    ]

    for text in test_texts:
        print(replace_num2words(text))
        print()

    test_texts = [
        '12:12',
        'с 12:12',
        'с 23:41',
        '23:41',
        'с 11:43',
        '11:43',
        'с 20:07',
        '20:07'
    ]

    for text in test_texts:
        print(re.sub(TIME_P, time_to_words, text))

    test_s = [
        '12 часов 12 минут',
        'с 12 часов 12 минут',
        'с 23 часов 41 минуты',
        '23 часа 41 минута',
        'с 11 часов 43 минут',
        '11 часов 43 минуты',
        'с 20 часов 7 минут',
        '20 часов 7 минут'
    ]

    for s in test_s:
        print(re.sub(TIME_WORDS_P, time_words_to_words, s))

    test_texts = [
        'время 23 часа 31 минута',
        'Платформа 281 км.',
        'Платформа 281-ый км.',
        'С 3-го вижу белый, выезжаю за на 1-й главный М6, 5949.',
        '57-ью вагонами'
    ]

    for text in test_texts:
        print(reg_to_asr(text))

