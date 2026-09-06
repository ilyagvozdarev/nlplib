import re


def tagging_by_regex(
    text: str, 
    tags_examples : dict[str, list[str]]
) -> str:
    """
    Inserts NER tags into the text to the right of entities matching the 
    corresponding tag (entities are defined via regex patterns)

    Parameters
    ----------
    text:
        input text
    tags_examples:
        mapping ner-tag -> regex patterns of tag examples
        example:
        tags_examples = {
            'rzo': ['локомотив(а)?', 'поезд(а)?', 'путь', 'станцию'],
            'sen': ['машинист', 'ДНЦ']
        }

    Returns
    -------
    text with inserted tags    
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
    """
    Removes tags of the form [B-***], [I-***], [BI-***] and the square brackets
    around the words they belong to
    """
    return re.sub(
        r"\[(.+?)\]\[(B|I|BI)-\w+\]", 
        rf'\1', 
        text, 
        flags=re.IGNORECASE
    )


def extract_tags_from_text(
    text, 
    bi=True,
    splitter = str.split
):
    """
    Extracts all unique tags of the form [...][B-...] or [...][I-...] from the text
    Useful for building a tag set from tagged text

    Parameters
    ----------
    subentities:
        whether to also extract inner tags (I-***) in case
        the entity consists of multiple words - [...][BI-...]
    splitter:
        delimiter for splitting an entity into sub-entities (by default, splits on whitespace)
    """
    tags = set()
    for entity, BI_tag in re.findall(r'\[(.+?)\]\[((?:B|I|BI)-\w+)\]', text):
            BI, tag = BI_tag.split('-')
            tags.add('B-' + tag)
            if bi and BI == 'BI' and len(splitter(entity)) >= 2:
                tags.add('I-' + tag)
    return tags


def text_with_entities_tags_to_entities_tags(
    text, 
    tags_to_indices, 
    bi = True,
    subent_splitter = str.split
):
    """
    Builds a list of entities and their corresponding tags (model input data)
    from a text with tags and the entities they belong to (both tags and entities
    must be enclosed in []).
    Entities without a tag are assigned the tag 'O'
    
    Parameters
    ----------
    text:
        text with entities and tags
        Example: 
        [Поезд][BI-rzo] прибыл на [станцию][BI-rzo]. [Подготовьтесь][BI-act] к [отцепке][BI-act] [локомотива][BI-rzo]
    tags_to_indices:
        dictionary of tag indices
    bi:
        whether to also form sub-entities with inner tags (I-***) in case
        the entity consists of multiple words (sub-entities).
        Splitting into sub-entities is only performed for entities marked
        with the universal BI tag (i.e. not yet broken down into tokens); entities
        that already have a final B- or I- tag are not split again.
    subent_splitter:
        delimiter for splitting an entity into sub-entities (by default, splits on whitespace)

    Returns
    -------
    entities - entities
    ner_tags - entity tags (entities without a tag get the index of tag 'O')
    """
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
        if bi and BI == 'BI':
            subentities = subent_splitter(entity)
        else:
            subentities = []

        if not subentities:
            subentities = [entity]

        entities.extend(subentities)
        tags = ['B-' + tag] + ['I-' + tag] * (len(subentities) - 1)
        ner_tags.extend([tags_to_indices[t] for t in tags])
        if not_entity:
            entities.append(not_entity)
            ner_tags.append(tags_to_indices['O'])

    assert len(entities) == len(ner_tags)
    return entities, ner_tags