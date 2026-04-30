from collections.abc import Sequence
import copy, re, logging, os

import numpy as np
import pandas as pd
from scipy import optimize

from ..utils.algo import levenshtein_with_edits
from .DiarizationConfig import DiarizationConfig
from ..processing.preprocessing import (
    normalize_text
)
from ..training.timing import secAsMinutes


logging.root.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)



def diartext_to_text_and_spks(
    diar_text: str,
    DC: DiarizationConfig,
    skip_meaningless_speaker: bool = True,
) -> tuple[str, str]:
    '''
        извлекает текст и спикеров из диаризированного текста

        Parameters
        ----------
        diar_text:
            диаризированный текст ('*spk_token* текст спикера *spk_token* текст спикера ...')

        DC:
            конфиг диаризации с speaker_prefix, speaker_suffix используемых
            для определения что слово является токеном спикера
        
        skip_meaningless_speaker:
            нужно ли пропускать (заменять на прошлого спикера) невалидные 
            (пустые или не циферные) имена спикеров. Если False - то будет исключение
        
        Notes
        -----
        - имена спикеров должны быть целыми числами
        - если слово начинается с speaker_prefix но не заканчивается speaker_suffix,
          то speaker_suffix добавляется к слову
        - если имя спикера в токене спикера является невалидным (не целое число), 
          то считается что спикер слова - прошлый спикер.
        - если у первого слова спикер отсутствует, то считается что спикер имеет имя '1'
        - диаризированный текст вначале разбивается по пробельному символу методом split(),
          поэтому в итоговом тексте все непрерывные последовательности пробельных
          символов будут заменены на 1 пробел (начальные и конечные пробелы будут отсутствовать)
    '''

    spk = "1"
    previous_spk = "1"
    result_text = []
    result_spk = []
    for word in diar_text.split():
        if word.startswith(DC.speaker_prefix):
            if not word.endswith(DC.speaker_suffix):
                word += DC.speaker_suffix
            spk = word[len(DC.speaker_prefix):-len(DC.speaker_suffix)]
            # Handle undefined behaviors of non-recognizable spk with a placeholder.
            try:
                spk_int = int(spk)
                if not spk:
                    raise ValueError("Seeing unexpected word: ", word)
                previous_spk = spk
            except ValueError as exc:
                if skip_meaningless_speaker:
                    print("Skipping meaningless speaker token:", word)
                    spk = previous_spk
                else:
                    raise exc
        else:
            result_text.append(word)
            result_spk.append(spk)
    return " ".join(result_text), " ".join(result_spk)
    

def build_speaker_token(DC: DeprecationWarning, speaker: str):
    return DC.speaker_prefix + speaker + DC.speaker_suffix


def text_and_spks_to_diartext(
    DC: DiarizationConfig,
    word_labels: list[str],
    speaker_labels: list[str],
) -> str:
    '''
        создает диаризированный текст ('*spk_token* текст спикера *spk_token* текст спикера ...'),
        из списка слов и списка спикеров

        Parameters
        ----------
        word_labels
        speaker_labels
        DC:
            конфиг диаризации с speaker_prefix, speaker_suffix используемых
            для создания токена спикера

        Returns
        -------
        diarized_text - диаризированный текст

        Notes
        -----
        - между любым спикером и текстом всегда вставляется пробел
    '''

    diarized_text = ''
    previous_speaker = None
    for word, speaker in zip(word_labels, speaker_labels):
        if speaker != previous_speaker:
            diarized_text += ' ' + build_speaker_token(DC, speaker)
        diarized_text += ' ' + word
        previous_speaker = speaker
    diarized_text = diarized_text.lstrip()

    return diarized_text



def diartext_to_diar_segments(
    diarized_text: str,
    DC: DiarizationConfig,
) -> list[dict]:
    '''
        todo протестировать метод
        преобразует диаризированный текст в сегменты диаризации

        Parameters
        ----------
        diarized_text:
            диаризированный текст ('*spk_token* текст спикера *spk_token* текст спикера ...')

        DC:
            конфиг диаризации с speaker_prefix, speaker_suffix используемых
            для определения что токен является токеном спикера

        Returns
        -------
        list[dict]:
            сегменты диаризации - список словарей вида:
            {
                "speaker": спикер,
                "text": текст
            }

    '''
    # ? в '.+?' делает .+ non-greedy
    spk_pattern = f'({DC.speaker_prefix}.+?{DC.speaker_suffix})'
    texts_spks = re.split(spk_pattern, diarized_text)
    if texts_spks[0] != '':
        raise ValueError("diarized_text starting not from speaker token")
    texts_spks = texts_spks[1:]
    spks = texts_spks[::2]
    texts = texts_spks[1::2]
    texts = [text.strip() for text in texts]
    return [{'speaker': spk, 'text': text} for spk, text in zip(spks, texts)]


def diar_segments_to_diartext(
    diar_segs: list, 
    spk_field = 'speaker',
    text_field = 'text',
    speaker_prefix = "<speaker:",
    speaker_suffix = ">"
):
    return ' '.join([
        (
            segment[spk_field] 
            if not speaker_prefix 
            else speaker_prefix + segment[spk_field] + speaker_suffix
        ) + ' ' + segment[text_field] 
        for segment in diar_segs
    ])


def diar_segments_to_diartext_text_spks_as_dict(segments, DC):
    diartext = diar_segments_to_diartext(segments)
    text, spks = diartext_to_text_and_spks(diartext, DC)
    return {
        'diarized_text': diartext, 
        'text': text, 
        'spk': spks,
    }


def group_diar_segments_by_speaker(segments) -> dict:
    '''
        группирует тексты сегментов диаризации по спикерам
    '''
    return pd.DataFrame(segments).groupby('speaker')['text'].apply(list).to_dict()
       

def export_diar_segments_to_txt(segments, path_txt, overwrite=True):
    '''
        экспорт сегментов диаризации в txt, в отдельных строках для каждого сегмента:
            спикер
            время начала сегмента
            время конца сегмента
            текст
        если overwrite=False то данные не будут перезаписываться если файл уже существует 
    '''
    if overwrite or not overwrite and not os.path.isfile(path_txt):
        with open(path_txt, 'w') as file_txt:
            for segment in segments:
                start_m_s = secAsMinutes(segment['start'])
                end_m_s = secAsMinutes(segment['end'])
                file_txt.write(f"{segment['speaker']}\n{start_m_s}\n{end_m_s}\n{segment['text']}\n\n")


def load_diar_segments_from_txt(path_txt):
    '''
        загрузка сегментов диаризации из txt, в отдельных строках для каждого сегмента:
            спикер
            время начала сегмента (может отсутствовать)
            время конца сегмента (может отсутствовать)
            текст
        в список сегментов вида {"speaker": str, "start": str, "end": str, "text": str}
    '''
    with open(path_txt) as f_txt:
        diar_segments = []
        segments = re.split(r'\n\n+', f_txt.read().strip())
        for segment in segments:
            lines = segment.split('\n')
            assert len(lines) == 4 or len(lines) == 2
            if len(lines) == 4:
                fields = ['speaker', 'start', 'end', 'text']
            elif len(lines) == 2:
                fields = ['speaker', 'text']
            else:
                raise Exception(
                    'segment length must be 4 ("speaker", "start", "end", "text")' + \
                    'or 2 ("speaker", "text")'
                )
            diar_segments.append(dict(zip(fields, lines)))

    return diar_segments


### генерация примеров

class TrainFields:
    def __init__(
        self,
        flavor = None,
        text_field = None,
        input_speaker_field = None,
        target_speaker_field = None
    ):
        '''
            flavor - тип примера {'HYP2ORA', 'DEG2REF'}
            text_field - поле с текстом
            input_speaker_field - поле с входными спикерами
            target_speaker_field - поле с целевыми спикерами
        '''
        self.flavor = flavor
        self.text_field = text_field
        self.input_speaker_field = input_speaker_field
        self.target_speaker_field = target_speaker_field


def get_generator_prompt_target_examples():
    def generator_prompt_target_examples(DC, utts, train_fields):
        for example in generate_prompt_target_examples(DC, utts, train_fields):
            yield example
    return generator_prompt_target_examples


def generate_prompt_target_examples(DC, utts, train_fields: TrainFields):
    '''
        генерирует обучающие примеры:
            все ключи-значения диаризации utt
            flavor - тип примера {'HYP2ORA', 'DEG2REF'}
            prompt - диаризированный текст промпта
            target - целевой диаризированный текст
    '''
    for utt in utts:
        words, p_speakers, t_speakers = get_data_for_utt(utt, train_fields)
        for prompt, target in generate_prompt_target_from_range_ver2(
            DC,
            words, p_speakers, t_speakers, 
            start=0, end=len(words), 
            train_fields=train_fields
        ):
            yield {
                **utt, 
                "flavor": train_fields.flavor, 
                "prompt": prompt, 
                "target": target
            }


def get_data_for_utt(utt, train_fields: TrainFields):
    '''
        возвращает список слов (текст поля text_field разделенный по пробельному символу),
        список входных спикеров и целевых спикеров задаваемых полями train_fields
        в диаризации utt
    '''
    t_speakers = []
    if train_fields.target_speaker_field in utt:
        t_speakers = utt[train_fields.target_speaker_field].split()

    return (
        utt[train_fields.text_field].split(), 
        utt[train_fields.input_speaker_field].split(),
        t_speakers
    )




def generate_prompt_target_from_range_ver2(
    DC: DiarizationConfig,
    words, p_speakers, t_speakers, 
    start, end,
    train_fields: TrainFields
):
    '''
        todo протестировать сравнив с generate_prompt_target_from_range
        формирует пары (prompt, target): 
        prompt (промпт) - диаризированный текст со спикерами p_speakers и словами 
        words + префикс и суффикс промпта, 
        target (завершение) - диаризированный текст со спикерами t_speakers и 
        словами words + суффикс завершения.

        Parameters
        ----------
        words - список слов
        p_speakers - спикеры для prompt
        t_speakers - спикеры для target

        DC:
            конфиг диаризации в котором используются:
            - speaker_prefix, speaker_suffix для создания токена спикера 
              (в text_and_spks_to_diartext)
            - префикс и суффикс промпта, суффикс завершения
            - максимальная длина промпта и завершения

        Notes
        -----
        Если длина промпта больше DC.EMIT_INPUT_LENGTH или длина завершения больше 
        DC.EMIT_TARGET_LENGTH то входные данные рекурсивно делятся на равные части 
        по середине получая 2 примера.
        Проверка и разбиение происходит 2 раза - для оцененной длины промпта и 
        завершения где не учитываются длины токенов спикеров, только слова текста и 
        для длины сформированных промпта и завершения.
    '''
    
    def generate_prompt_target_halves(
            words, 
            p_speakers, t_speakers, 
            start, end, 
            train_fields
    ):
        yield from generate_prompt_target_from_range_ver2(
            words, p_speakers, t_speakers, 
            start, (start + end) // 2, 
            train_fields=train_fields
        )
        yield from generate_prompt_target_from_range_ver2(
            words, p_speakers, t_speakers, 
            (start + end) // 2, end, 
            train_fields=train_fields
        )


    PROMPT_PREFIX = DC.PROMPT_PREFIX
    PROMPT_SUFFIX = DC.PROMPT_SUFFIX
    COMPLETION_SUFFIX = DC.COMPLETION_SUFFIX
    EMIT_INPUT_LENGTH = DC.EMIT_INPUT_LENGTH
    EMIT_TARGET_LENGTH = DC.EMIT_TARGET_LENGTH

    estimated_prompt_length = (
        len(PROMPT_PREFIX)
        + len(" ".join(words[start:end]))
        + len(PROMPT_SUFFIX)
    )
    # print('estimated_prompt_length = ', estimated_prompt_length)
    # print('len(" ".join(words[start:end]) = ', len(" ".join(words[start:end])))
    # print('EMIT_INPUT_LENGTH = ', EMIT_INPUT_LENGTH)
    # print('EMIT_TARGET_LENGTH = ', EMIT_TARGET_LENGTH)

    if (
        estimated_prompt_length > EMIT_INPUT_LENGTH
        or estimated_prompt_length > EMIT_TARGET_LENGTH
    ):
        yield from generate_prompt_target_halves(
            words, 
            p_speakers, t_speakers, 
            start, end, 
            train_fields
        )
        return

    prompt = PROMPT_PREFIX
    target = ""

    prompt += text_and_spks_to_diartext(DC, words[start:end], p_speakers[start:end])
    prompt += PROMPT_SUFFIX

    if train_fields.target_speaker_field:
        target += text_and_spks_to_diartext(DC, words[start:end], t_speakers[start:end])
        target += COMPLETION_SUFFIX

    # print('target = ', len(target))
    # print('prompt = ', len(prompt))
    
    if (
        len(prompt) <= EMIT_INPUT_LENGTH
        and len(target) <= EMIT_TARGET_LENGTH
    ):
        yield (prompt, target)
    else:
        yield from generate_prompt_target_halves(
            words, 
            p_speakers, t_speakers, 
            start, end, 
            train_fields
        )


# todo удалить после тестирования generate_prompt_target_from_range_ver2
def generate_prompt_target_from_range(
    DC: DiarizationConfig,
    words, p_speakers, t_speakers, 
    start, end,
    train_fields: TrainFields
):
    """
    Если длина промпта больше diar_config.EMIT_INPUT_LENGTH или diar_config.EMIT_TARGET_LENGTH или
    длина target больше diar_config.EMIT_TARGET_LENGTH
    то рекурсивно делим входные данные на равные части по середине (2 примера)
    """

    PROMPT_PREFIX = DC.PROMPT_PREFIX
    PROMPT_SUFFIX = DC.PROMPT_SUFFIX
    COMPLETION_SUFFIX = DC.COMPLETION_SUFFIX
    EMIT_INPUT_LENGTH = DC.EMIT_INPUT_LENGTH
    EMIT_TARGET_LENGTH = DC.EMIT_TARGET_LENGTH

    estimated_prompt_length = (
        len(PROMPT_PREFIX)
        + len(" ".join(words[start:end]))
        + len(PROMPT_SUFFIX)
    )
    # print('estimated_prompt_length = ', estimated_prompt_length)
    # print('len(" ".join(words[start:end]) = ', len(" ".join(words[start:end])))
    # print('EMIT_INPUT_LENGTH = ', EMIT_INPUT_LENGTH)
    # print('EMIT_TARGET_LENGTH = ', EMIT_TARGET_LENGTH)
    if (
        estimated_prompt_length > EMIT_INPUT_LENGTH
        or estimated_prompt_length > EMIT_TARGET_LENGTH
    ):
        yield from generate_prompt_target_from_range(
            words, p_speakers, t_speakers, 
            start, (start + end) // 2, 
            train_fields=train_fields
        )
        yield from generate_prompt_target_from_range(
            words, p_speakers, t_speakers, 
            (start + end) // 2, end, 
            train_fields=train_fields
        )
        return

    prompt = PROMPT_PREFIX
    previous_p_spk = ""
    target = ""
    previous_t_spk = ""

    for i in range(start, end):
        word = words[i]
        p_spk = p_speakers[i]
        if p_spk != previous_p_spk:
            if previous_p_spk:
                prompt += " "
            prompt += build_speaker_token(DC, p_spk)
        prompt += " " + word
        previous_p_spk = p_spk

        if train_fields.target_speaker_field:
            t_spk = t_speakers[i]
            if t_spk != previous_t_spk:
                if previous_t_spk:
                    target += " "
                target += build_speaker_token(DC, t_spk)
            target += " " + word
            previous_t_spk = t_spk

    prompt += PROMPT_SUFFIX
    target += COMPLETION_SUFFIX

    # print('target = ', len(target))
    # print('prompt = ', len(prompt))
    
    if (
        len(prompt) <= EMIT_INPUT_LENGTH
        and len(target) <= EMIT_TARGET_LENGTH
    ):
        yield (prompt, target)
    else:
        yield from generate_prompt_target_from_range(
            words, p_speakers, t_speakers, 
            start, (start + end) // 2, 
            train_fields=train_fields
        )
        yield from generate_prompt_target_from_range(
            words, p_speakers, t_speakers, 
            (start + end) // 2, end, 
            train_fields=train_fields
        )


def get_aligned_speakers(
    src_text: str,
    tgt_text: str,
    src_spk: str,
    print_debug_info: bool = False,
) -> str:
  '''
    применяет к src_spk выравнивание src_text -> tgt_text:
    для отсутствующих в src_text токенов (которые есть в tgt_text) вставляет spk=-1
  '''

  num_insertions, num_deletions = 0, 0

  # выравнивание
  _, align = levenshtein_with_edits(
      normalize_text(src_text), 
      normalize_text(tgt_text)
  )

  src_spk_list = src_spk.split()
  src_spk_align = []

  # применяем выравнивание к tgt speakers
  for i, j in align:
    if i == -1:
      # src has insertion
      src_spk_align.append("-1")
      num_insertions += 1
    elif j == -1:
      # src has deletion
      num_deletions += 1
      continue
    else:
      src_spk_align.append(src_spk_list[i])

  src_spk_align = " ".join(src_spk_align)

  if print_debug_info:
    print("Number of insertions: ", num_insertions)
    print("Number of deletions: ", num_deletions)
    # This is not the traditional denominator of WER. Instead, this is
    # len(src) + len(tgt) - len(SUB).
    print("Length of align pairs: ", len(align))

  return src_spk_align


def get_oracle_speakers(ref_spk: str, hyp_spk_align: str) -> Sequence[int]:
  '''
    формирует oracle спикеров по эталонным спикерам для выровненных спикеров гипотезы:
    - находим переименование спикеров гипотезы на имена спикеров эталона по переназначению 
      полученным Венгерским алгоритмом
    - для пропущенных слов которые отсутствуют в гипотезе но есть в эталоне оставляем спикеров из эталона 
      (если спикер пропущен и в эталоне то берем спикера предыдущего слова эталона,
      если это первое слово то спикер = 1),
      для общих (совпадающих или замененных) слов назначаем спикера гипотезы но переименовываем
      в соответствии с найденным переименованием.


    Parameters
    ----------
    ref_spk:
      спикеры эталона (строки представляют int > 0)

    hyp_spk_align:
      выровненные спикеры гипотезы (для отсутствующих слов спикер -1)
      (строки представляют int > 0)
  '''

  ref_spk_list = [int(x) for x in ref_spk.split()]
  hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

  # print('ref_spk_list = ', ref_spk_list)
  # print('hyp_spk_align_list = ', hyp_spk_align_list)

  # cost matrix как часто у одного и того же слова встречается спикер i и j
  max_spk = max(max(ref_spk_list), max(hyp_spk_align_list))
  cost_matrix = np.zeros((max_spk, max_spk))
  for aligned, original in zip(hyp_spk_align_list, ref_spk_list):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment.
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )

  # Build oracle.
  spk_oracle = ref_spk_list.copy()
  for i in range(len(ref_spk_list)):
    if hyp_spk_align_list[i] == -1:
      # пропущенное слово, вставляем для него спикера из эталона
      if ref_spk_list[i] == -1:
        # если спикер пропущен и в эталоне то берем спикера предыдущего слова эталона,
        # если это первое слово то спикер = 1
        if i == 0:
          spk_oracle[i] = 1
        else:
          spk_oracle[i] = spk_oracle[i - 1]
      continue
    # print('hyp_spk_align_list[i] - 1 = ', hyp_spk_align_list[i] - 1)
    # print('row_index = ', row_index)
    assert hyp_spk_align_list[i] - 1 >= 0
    assert row_index[hyp_spk_align_list[i] - 1] == hyp_spk_align_list[i] - 1
    spk_oracle[i] = col_index[hyp_spk_align_list[i] - 1] + 1

  return spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def TPST(
    src_text: str, 
    src_spk: str, 
    tgt_text: str, 
    tgt_spk: str
) -> str:
  '''
    перенос спикеров src_spk текста src_text в текст tgt_text другой диаризации со спикерами tgt_spk.
  '''

  if len(tgt_text.split()) != len(tgt_spk.split()):
    raise ValueError("tgt_text and tgt_spk must have the same length")
  if len(src_text.split()) != len(src_spk.split()):
    raise ValueError("src_text and src_spk must have the same length")
  
  tgt_spk_align = get_aligned_speakers(
      tgt_text=tgt_text, 
      src_text=src_text,
      src_spk=src_spk,
  )
  oracle_speakers = get_oracle_speakers(
      ref_spk=tgt_spk, 
      hyp_spk_align=tgt_spk_align
  )
  return " ".join([str(x) for x in oracle_speakers])


# We can use this to finetune LLM.
# Inputs (prompts): hyp diarized text
# Targets: hyp diarized text with oracle speakers
# Если тексты ref и hyp не отличаются то результат равен ref_spk
def spk_oracle(json_dict: dict[str, str]) -> str:
  """Apply reference speakers to hypothesis."""
  return TPST(
      src_text=json_dict["ref_text"],
      src_spk=json_dict["ref_spk"],
      tgt_text=json_dict["hyp_text"],
      tgt_spk=json_dict["hyp_spk"],
  )


# Similar to ref_to_oracle, but the opposite direction.
# We can use this to finetune LLM.
# Inputs (prompts): ref diarized text with degraded speakers
# Targets: ref diarized text
# Если тексты ref и hyp не отличаются то результат равен hyp_spk
def spk_degraded(json_dict: dict[str, str]) -> str:
  """Apply hypothesis speakers to reference."""
  return TPST(
      src_text=json_dict["hyp_text"],
      src_spk=json_dict["hyp_spk"],
      tgt_text=json_dict["ref_text"],
      tgt_spk=json_dict["ref_spk"],
  )


def truncate_suffix_and_tailing_text(text: str, suffix: str) -> str:
  """Tailing text after suffix should be removed as well."""
  if suffix and suffix in text:
    return text[: text.find(suffix)]
  return text



def merge_speakers(segments: list[dict]):
    '''
      Соединение сегментов диаризации, где говорит один и тот же спикер.
      Если спикер отсутствует, то ему назначется имя "SPEAKER_XX"

      Parameters
      ----------
      segments:
        список сегментов вида:
        {
          "speaker": спикер,
          "text": текст,
          "start": момент начала текста сегмента (количество секунд от начала аудио),
          "end": момент конца текста сегмента (количество секунд от начала аудио),
        }
    ''' 
    
    if not segments:
        return None
    
    merged_segments = []
    cur_segment = segments[0]

    curr_spk = cur_segment["speaker"] if cur_segment.get("speaker") else "SPEAKER_XX"
    text = cur_segment["text"]
    start = cur_segment["start"]
    end = cur_segment["end"]

    for cur_segment in segments[1:]:
        spk = cur_segment["speaker"] if cur_segment.get("speaker") else "SPEAKER_XX"

        if curr_spk != spk:
            merged_segments.append({
                "speaker": curr_spk,
                "text": text,
                "start": start,
                "end": end
            })
            text = cur_segment["text"]
            start = cur_segment["start"]
            end = cur_segment["end"]
        else:
            text += " " + cur_segment["text"].strip()
            end = cur_segment["end"]

        curr_spk = spk

    merged_segments.append({
        "speaker": curr_spk,
        "text": text,
        "start": start,
        "end": end
    })
    return merged_segments


def diar_segments_speakers_to_ids(
      diar_segms, 
      DC: DiarizationConfig = None
):
    '''
      преобразует имена спикеров сегментов в индексы спикеров.
      если задан конфиг диаризации, то добавляет из него префикс и суффикс спикера

      Parameters
      ----------
      diar_segms:
          список сегментов вида:
          {
            "speaker": спикер,
            ...
          }
      DC:
          конфиг диаризации
    '''
    diar_segms_res = copy.deepcopy(diar_segms)
    speakers = [segment['speaker'] for segment in diar_segms_res]
    speakers_ids = speakers_to_ids(speakers)

    for segment, spk_id in zip(diar_segms_res, speakers_ids):
        spk = str(spk_id)
        if DC:
           spk = DC.speaker_prefix + spk + DC.speaker_suffix
        segment['speaker'] = spk
    
    return diar_segms_res


def speakers_to_ids(speakers: Sequence[str]) -> list[str]:
  """
    преобразует последовательность имен спикеров в индексы спикеров от 1
  """
  spk_to_id = dict(zip(set(speakers), range(1, len(speakers) + 1)))
  return [str(spk_to_id[spk]) for spk in speakers]
