import re
from functools import partial

def tagging_by_regex(
    text: str, 
    tags_examples : dict[str, list[str]]
) -> str:

    """
    вставляет в тексте ner-тэги справа от сущностей соответствующего тэга (сущности задаются по regex-шаблону)

    Parameters
    ----------
    text:
        входной текст

    tags_examples:
        словарь примеров тэгов
        key: ner-тэг
        value: regex-шаблоны примеров тэга 

        пример:
        tags_examples = {
            'rzo': ['локомотив(а)?', 'поезд(а)?', 'путь', 'станцию'],
            'sen': ['машинист', 'ДНЦ']
        }

    Returns
    -------
    str:
    текст с вставленными тэгами    

    """

    for tag, tag_examples in tags_examples.items():
        examples_s = f"({'|'.join(tag_examples)})"
        search_pattern = rf'(?<!\[){examples_s}(?!\](\[(B|I)|\w))'
        tag_pattern = rf'[\1][BI-{tag}]'
        text = re.sub(
            search_pattern, 
            tag_pattern, 
            text, 
            flags=re.IGNORECASE
        )

    return text


def remove_tags(text):
    '''
        удаляет тэги вида [B-***], [I-***], [BI-***] и квадратные скобки у слов 
        к которым они относятся
    '''
    return re.sub(
        r"\[(.+?)\]\[(B|I|BI)-\w{,10}\]", 
        rf'\1', 
        text, 
        flags=re.IGNORECASE
    )


def insert_space_after_tags(text):
    '''
        вставляем пробел после ner-тэга если сразу после него непробельный символ (как правило пунктуация). 
        Используется при формировании индексов тэгов токенов, чтобы при сплите по пробелу сущности 
        были только со своим тегом

        Notes
        -----
        нет необходимости использовать этот метод при формировании списка слов и тэгов 
        с помощью text_with_tags_to_words_tags 
    '''
    return re.sub(r'(\[[(?:B|I|BI)]\S+?\])(\S)', r'\1 \2', text)


def extract_tags_from_text(
    text, 
    also_subentities=True,
    subentities_splitter = str.split
):
    '''
        извлекает все уникальные тэги вида B-*** или I-*** из текста
        (в тексте должны быть заключены в [], то есть например [B-***]).
        Полезно для формирования tag set по размеченному тексту

        also_subentities:
            нужно ли также извлекать inner тэги (I-***) в случае
            если сущность состоит из нескольких слов (подсущностей)
        subentities_splitter:
            разделитель сущности на подсущности (по умолчанию по последовательности пробельных символов)
    '''
    tags = set()
    for entity, BI_tag in re.findall(r'\[(.+?)\]\[((?:B|I|BI)-\w{,10})\]', text):
            BI, tag = BI_tag.split('-')
            tags.add('B-' + tag)
            if also_subentities and BI == 'BI' and len(subentities_splitter(entity)) >= 2:
                tags.add('I-' + tag)
    return tags


def text_with_entities_tags_to_entities_tags(
    text, 
    tags_to_indices, 
    also_subentities = True,
    subentities_splitter = str.split
):
    '''
        по тексту с тэгами и сущностями (тэги и сущности должны быть заключены в []) к 
        которым они относятся формирует список сущностей и список соответствующих им 
        тэгов (входные данные модели).
        Сущностям без тэга назначается тэг 'O'
        
        Parameters
        ----------
        text:
            текст с сущностями и тэгами
            Пример: 
            [Поезд][BI-rzo] прибыл на [станцию][BI-rzo]. [Подготовьтесь][BI-act] к [отцепке][BI-act] [локомотива][BI-rzo]
        tags_to_indices:
            словарь индексов тэгов
        also_subentities:
            нужно ли также формировать подсущности с inner тэгами (I-***) в случае
            если сущность состоит из нескольких слов (подсущностей)
        subentities_splitter:
            разделитель сущности на подсущности (по умолчанию по последовательности пробельных символов)

        Returns
        -------
        entities - сущности
        ner_tags - тэги сущностей (у сущностей без тэга индекс тэга 'O')
    '''
    entity_with_tag_pattern = r'\[(.+?)\]\[((?:B|I|BI)-\w+)\]'

    entities = []
    ner_tags = []

    parts = re.split(entity_with_tag_pattern, text)

    if len(parts) == 1:
        # no tags
        return [text], [tags_to_indices['O']]

    if len(parts) >= 3:
        if parts[0]:
            entities.append(parts[0])
            ner_tags.append(tags_to_indices['O'])
        parts = parts[1:]

    for entity, BI_tag, not_entity in zip(parts[::3], parts[1::3], parts[2::3]):
        BI, tag = BI_tag.split('-')
        if not also_subentities:
            entities.append(entity)
            ner_tags.append(tags_to_indices['B-' + tag])
        else:
            subentities = subentities_splitter(entity)
            entities.extend(subentities)
            tags = ['B-' + tag] + ['I-' + tag] * (len(subentities) - 1)
            ner_tags.extend([tags_to_indices[tag] for tag in tags])
        if not_entity:
            entities.append(not_entity)
            ner_tags.append(tags_to_indices['O'])

    assert len(entities) == len(ner_tags)

    return entities, ner_tags