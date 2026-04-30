"""A Python implementation of the ASR and diarization metrics.

We implement:
- Word Error Rate (WER):  This is the most wfileely used metrics for
  Automatic Speech Recognition (ASR). See
  https://en.wikipedia.org/wiki/Word_error_rate
- Word Diarization Error Rate (WDER): This metric was proposed by Google. See
  Shafey, Laurent El, Hagen Soltau, and Izhak Shafran. "Joint speech
  recognition and speaker diarization via sequence transduction."
  arXiv preprint arXiv:1907.05337 (2019).
  https://arxiv.org/pdf/1907.05337
- Concatenated minimum-permutation Word Error Rate (cpWER): This metric was
  used in the CHiME-6 Challenge. See
  Watanabe, Shinji, et al. "CHiME-6 challenge: Tackling multispeaker speech
  recognition for unsegmented recordings."
  arXiv preprint arXiv:2004.09249 (2020).
  https://arxiv.org/pdf/2004.09249

Note: This implementation is different from Google's internal implementation
that we used in the paper, but is a best-effort attempt to replicate the
results. The biggest differences are from text normalization, such as
de-punctuation.
"""

import dataclasses
from typing import Any, Optional
import numpy as np
from scipy import optimize
import tqdm
import logging
from ..processing.preprocessing import normalize_text
from ..utils import levenshtein_with_edits
from .utils import TPST
from sklearn.metrics import classification_report
from sklearn.metrics.cluster import v_measure_score, rand_score


@dataclasses.dataclass
class UtteranceMetrics:
  """Metrics for one utterance."""

  # wer_total: wer_correct + wer_sub + wer_delete (знаменатель WER)
  # wer: WER
  wer_insert: int = 0
  wer_delete: int = 0
  wer_sub: int = 0
  wer_correct: int = 0
  wer_total: int = 0      
  wer: int = 0            

  # wder_sub: количество слов (замен или правильных) с неверными токенами говорящего (числитель WDER)
  # wder_correct: количество слов (замен или правильных) с верными токенами говорящего (нет в формуле WDER)
  # wder_total: знаменатель WDER
  # wder: WDER
  wder_sub: int = 0       
  wder_correct: int = 0   
  wder_total: int = 0     
  wder: int = 0          

  # WER но для конкретного назначения спикеров
  # Важно:
  # Итоговый metrics.cpwer после суммирования по всем назначениям не равен
  # (metrics.cpwer_insert + metrics.cpwer_delete + metrics.cpwer_sub) / metrics.cpwer_total
  # так как каждая из этих метрик это сумма всех отдельных соответствующих метрик по каждому назначению 
  # то есть: 
  #   metrics.cpwer = sum((metrics.cpwer_insert + metrics.cpwer_delete + metrics.cpwer_sub) / metrics.cpwer_total)
  #                != (sum(metrics.cpwer_insert) + sum(metrics.cpwer_delete) + sum(metrics.cpwer_sub)) / sum(metrics.cpwer_total)
  cpwer_insert: int = 0   
  cpwer_delete: int = 0
  cpwer_sub: int = 0
  cpwer_correct: int = 0
  cpwer_total: int = 0
  cpwer: int = 0


def merge_cpwer(
    wer_metrics: list[UtteranceMetrics], metrics: UtteranceMetrics
) -> None:
  """Compute cpWER metrics by merging a list of WER metrics."""
  for utt in wer_metrics:
    metrics.cpwer_insert += utt.wer_insert
    metrics.cpwer_delete += utt.wer_delete
    metrics.cpwer_sub += utt.wer_sub
    metrics.cpwer_correct += utt.wer_correct
    metrics.cpwer_total += utt.wer_total
    metrics.cpwer += utt.wer


def compute_wer(
    hyp_text: str, ref_text: str
) -> tuple[UtteranceMetrics, list[tuple[int, int]]]:
  """Compute the word error rate of an utterance."""
  result = UtteranceMetrics()
  hyp_normalized = normalize_text(hyp_text)
  ref_normalized = normalize_text(ref_text)
  hyp_words = hyp_normalized.split()
  ref_words = ref_normalized.split()

  # Get the alignment.
  _, align = levenshtein_with_edits(ref_normalized, hyp_normalized)

  # Apply the alignment on ref speakers.
  for i, j in align:
    if i == -1:
      result.wer_insert += 1
    elif j == -1:
      result.wer_delete += 1
    else:
      if ref_words[i] == hyp_words[j]:
        result.wer_correct += 1
      else:
        result.wer_sub += 1

  result.wer_total = result.wer_correct + result.wer_sub + result.wer_delete
  result.wer = (
    result.wer_sub + result.wer_delete + result.wer_insert
  ) / result.wer_total 
  assert result.wer_total == len(ref_words)
  return result, align


def compute_utterance_metrics(
    hyp_text: str,
    ref_text: str,
    hyp_spk: Optional[str] = None,
    ref_spk: Optional[str] = None,
) -> UtteranceMetrics:
  """Compute all metrics of an utterance."""
  hyp_normalized = normalize_text(hyp_text)
  ref_normalized = normalize_text(ref_text)
  hyp_words = hyp_normalized.split()
  ref_words = ref_normalized.split()

  ########################################
  # Compute WER.
  ########################################
  result, align = compute_wer(hyp_text, ref_text)

  compute_diarization_metrics = hyp_spk or ref_spk
  if not compute_diarization_metrics:
    return result

  if not (hyp_spk and ref_spk):
    logging.info(f"hyp_spk and ref_spk is unset")

  ########################################
  # Compute WDER.
  ########################################
  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  ref_spk_list = [int(x) for x in ref_spk.split()]

  # print('hyp_spk_list = ', hyp_spk_list)
  # print('ref_spk_list = ', ref_spk_list)

  if len(hyp_spk_list) != len(hyp_words):
    raise ValueError("hyp_spk and hyp_text must have the same length.")
  if len(ref_spk_list) != len(ref_words):
    raise ValueError("ref_spk and ref_text must have the same length.")
  hyp_spk_list_align = []
  ref_spk_list_align = []

  # собираем только spk из hyp и ref соответствующих общим словам (заменам и правильным словам)
  for i, j in align:
    if i != -1 and j != -1:
      ref_spk_list_align.append(ref_spk_list[i])
      hyp_spk_list_align.append(hyp_spk_list[j])

  # Build cost matrix (cost_matrix[i,j] = количество раз когда у слова в эталоне спикер 
  # i, а в гипотезе j)
  # max_spk - максимальный номер спикера среди общих слов
  max_spk = max(max(ref_spk_list_align), max(hyp_spk_list_align))
  cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
  for aligned, original in zip(ref_spk_list_align, hyp_spk_list_align):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment (находим максимизирующее назначение Венгерским алгоритмом 
  # чтобы сопоставить имена спикеров)
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )
  # result.wder_correct: 
  #   количество слов (замен или правильных) с верными токенами 
  #   говорящего (нет в формуле WDER)
  # result.wder_total:
  #   можно было и len(hyp_spk_list_align) так как длины равны (знаменатель WDER)
  # result.wder_sub:
  #   количество слов (замен или правильных) с неверными токенами говорящего (числитель WDER)

  result.wder_correct = int(cost_matrix[row_index, col_index].sum())  
  result.wder_total = len(ref_spk_list_align)  
  result.wder_sub = result.wder_total - result.wder_correct  
  result.wder = result.wder_sub / result.wder_total

  ########################################
  # Compute cpWER.
  ########################################
  spk_pair_metrics = {}
  cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
  for i in range(1, max_spk + 1):
    # все слова эталона сказанные спикером i
    # todo получается что считается только для спикеров номера которых меньше чем номера тех которые встречаются 
    # в общих словах (1..max_spk+1)
    # не является ошибкой? если спикер в hyp или ref встречается только в необщих словах и его номер больше максимального среди общих, 
    # то эти слова учитываться не будут в метрике, а если меньше максимального то будут.
    ref_words_for_spk = [
        ref_words[k] for k in range(len(ref_words)) if ref_spk_list[k] == i
    ]
    if not ref_words_for_spk:
      continue

    for j in range(1, max_spk + 1):
      # все слова гипотезы сказанные спикером j
      hyp_words_for_spk = [
          hyp_words[k] for k in range(len(hyp_words)) if hyp_spk_list[k] == j
      ]
      if not hyp_words_for_spk:
        continue
      # считаем WER по всем словам сказанным в гипотезе спикером i, а в эталоне спикером j
      spk_pair_metrics[(i, j)], _ = compute_wer(
          hyp_text=" ".join(hyp_words_for_spk),
          ref_text=" ".join(ref_words_for_spk),
      )
      # максимизируем по количеству одинаковых слов (wer_correct) сказанным в гипотезе спикером i, а в эталоне спикером j
      cost_matrix[i - 1, j - 1] = spk_pair_metrics[(i, j)].wer_correct

  # Solve alignment (находим максимизирующее количество одинаковых слов назначение Венгерским алгоритмом 
  # чтобы сопоставить имена спикеров)
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )
  # суммируем WER-метрики каждого полученного сопоставления
  metrics_to_concat = []
  for r, c in zip(row_index, col_index):
    if (r + 1, c + 1) not in spk_pair_metrics:
      continue
    # print('spk ref = ', r + 1, 'spk hyp = ', c + 1)
    # print(spk_pair_metrics[(r + 1, c + 1)].__dict__)
    # print()

    metrics_to_concat.append(spk_pair_metrics[(r + 1, c + 1)])
  merge_cpwer(metrics_to_concat, result)
  return result


def compute_classification_metrics(
    hyp_text: str,
    ref_text: str,
    hyp_spk: str,
    ref_spk: str
):
  hyp_spk_aligned = TPST(hyp_text, hyp_spk, ref_text, ref_spk)
  return classification_report(
    y_pred=hyp_spk_aligned.split(), 
    y_true=ref_spk.split(), 
    output_dict=True
  )


def compute_clusterization_metrics(
    hyp_text: str,
    ref_text: str,
    hyp_spk: str,
    ref_spk: str
):
  hyp_spk_aligned = TPST(hyp_text, hyp_spk, ref_text, ref_spk)
  return {
    'v_measure': v_measure_score(
       labels_pred=hyp_spk_aligned.split(), 
       labels_true=ref_spk.split()
     ),
     'rand_index': rand_score(
       labels_pred=hyp_spk_aligned.split(), 
       labels_true=ref_spk.split()
     )
	}


def compute_metrics_on_json_dict(
    diar_utts,
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk",
    compute_classification_clusterization_metrics = True
) -> dict[str, Any]:
  """Compute metrics for all utterances in a json object."""
  compute_diarization_metrics = ref_spk_field or hyp_spk_field
  if compute_diarization_metrics:
    if not (ref_spk_field and hyp_spk_field):
      raise ValueError(
          "hyp_spk_field and ref_spk_field must be both unset or both set."
      )
  
  result_dict = {
      "utterances": [],
  }
  for utt in tqdm.tqdm(diar_utts):
      # print(utt['file'])
      if not (utt[hyp_text_field] and utt[hyp_spk_field]):
          utt_result = {}
      else:
          if compute_diarization_metrics:
            utt_metrics = compute_utterance_metrics(
              hyp_text=utt[hyp_text_field],
              ref_text=utt[ref_text_field],
              hyp_spk=utt[hyp_spk_field],
              ref_spk=utt[ref_spk_field],
            )
          else:
            utt_metrics = compute_utterance_metrics(
              hyp_text=utt[hyp_text_field],
              ref_text=utt[ref_text_field],
            )
          utt_result = {'diarization_metrics': dataclasses.asdict(utt_metrics)}

      utt_result['file'] = utt['file']

      if compute_classification_clusterization_metrics:
          utt_result['classification_metrics'] = compute_classification_metrics(
              hyp_text=utt[hyp_text_field],
              ref_text=utt[ref_text_field],
              hyp_spk=utt[hyp_spk_field],
              ref_spk=utt[ref_spk_field]
          )
          utt_result['clusterization_metrics'] = compute_clusterization_metrics(
              hyp_text=utt[hyp_text_field],
              ref_text=utt[ref_text_field],
              hyp_spk=utt[hyp_spk_field],
              ref_spk=utt[ref_spk_field]
          )

      result_dict["utterances"].append(utt_result)
    

  final_wer_total = 0
  final_wer_correct = 0
  final_wer_sub = 0
  final_wer_delete = 0
  final_wer_insert = 0
  final_wder_total = 0
  final_wder_correct = 0
  final_wder_sub = 0
  final_cpwer_total = 0
  final_cpwer_correct = 0
  final_cpwer_sub = 0
  final_cpwer_delete = 0
  final_cpwer_insert = 0


  classification_metrics = {'precision_sum': 0, 'recall_sum': 0, 'f1-score_sum': 0, 'accuracy_sum': 0}
  clusterization_metrics = {'v_measure_sum': 0, 'rand_index_sum': 0}

  for utt in result_dict["utterances"]:

    # в utt только ключ file так как hyp_text_field или hyp_spk_field не заданы - 
    # метрики не посчитались
    if set(utt.keys()) == set(['file']):
      continue

    utt_diar = utt['diarization_metrics']
    utt_class = utt['classification_metrics']
    utt_clust = utt['clusterization_metrics']

    final_wer_total += utt_diar["wer_total"]
    final_wer_correct += utt_diar["wer_correct"]
    final_wer_sub += utt_diar["wer_sub"]
    final_wer_delete += utt_diar["wer_delete"]
    final_wer_insert += utt_diar["wer_insert"]
    if compute_diarization_metrics:
      final_wder_total += utt_diar["wder_total"]
      final_wder_correct += utt_diar["wder_correct"]
      final_wder_sub += utt_diar["wder_sub"]
      final_cpwer_total += utt_diar["cpwer_total"]
      final_cpwer_correct += utt_diar["cpwer_correct"]
      final_cpwer_sub += utt_diar["cpwer_sub"]
      final_cpwer_delete += utt_diar["cpwer_delete"]
      final_cpwer_insert += utt_diar["cpwer_insert"]
    if compute_classification_clusterization_metrics:
      classification_metrics['precision_sum'] += utt_class['macro avg']['precision']
      classification_metrics['recall_sum'] += utt_class['macro avg']['recall']
      classification_metrics['f1-score_sum'] += utt_class['macro avg']['f1-score']
      classification_metrics['accuracy_sum'] += utt_class['accuracy']
      clusterization_metrics['v_measure_sum'] += utt_clust['v_measure']
      clusterization_metrics['rand_index_sum'] += utt_clust['rand_index']

	
    

  result_dict["WER (micro)"] = (
      final_wer_sub + final_wer_delete + final_wer_insert
  ) / final_wer_total

  if compute_diarization_metrics:
    result_dict["WDER (micro)"] = final_wder_sub / final_wder_total
    result_dict["cpWER (micro)"] = (
        final_cpwer_sub + final_cpwer_delete + final_cpwer_insert
    ) / final_cpwer_total
    
  if compute_classification_clusterization_metrics:
    utts_count = len(result_dict["utterances"])
    result_dict["classification_metrics"] = {
      'accuracy_macro': classification_metrics['accuracy_sum'] / utts_count,
      'f1-score_macro': classification_metrics['f1-score_sum'] / utts_count,
      'recall_macro': classification_metrics['recall_sum'] / utts_count,
      'precision_macro': classification_metrics['precision_sum'] / utts_count
    }
    result_dict["clusterization_metrics"] = {
      'v_measure_macro': clusterization_metrics['v_measure_sum'] / utts_count,
      'rand_index_macro': clusterization_metrics['rand_index_sum'] / utts_count
    }


  return result_dict
