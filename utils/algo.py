"""
Levenshtein algorithm.

Note: This Python implementation is very inefficient. Please use this C++
implementation instead: https://github.com/wq2012/word_levenshtein
"""
import numpy as np
from enum import Enum


class EditOp(Enum):
  Correct = 0
  Substitution = 1
  Insertion = 2
  Deletion = 3


def levenshtein_with_edits(
    text1: str,
    text2: str,
    print_debug_info: bool = False
) -> tuple[int, list[tuple[int, int]]]:
  """
  Computes the Levenshtein edit distance and alignment between the
  strings text1 and text2
  (words in the strings must be separated by whitespace characters)

  Returns
  -------
  (edit_distance, alignment)
  alignment:
    an alignment from (0, 0) to (len(text1), len(text2))
    alignment elements:
    (-1, j) - insertion (in text2, transition from j to j+1)
    (i, -1) - deletion (in text1, transition from i to i+1)
    (i, j) - correct/substitution (transition from i, j to i+1, j+1)
  Example:
    for 'a b' and 'a c' the output would be:
    (1, [(0,0), (1,1)]
  """
  align = []
  s1 = text1.split()
  s2 = text2.split()
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1+1, n2+1), dtype=np.int32)
  backptr = np.zeros((n1+1, n2+1), dtype=EditOp)

  for i in range(n1+1):  # text1
    costs[i][0] = i  # deletions

  for j in range(n2):  # text2
    costs[0][j+1] = j+1  # insertions
    for i in range(n1):  # text1
      # calculate the cost of the operation for [i+1][j+1]
      # (i,j) <- (i,j-1)
      ins = costs[i+1][j] + 1
      # (i,j) <- (i-1,j)
      del_ = costs[i][j+1] + 1
      # (i,j) <- (i-1,j-1)
      sub = costs[i][j] + (s1[i] != s2[j])
      costs[i + 1][j + 1] = min(ins, del_, sub)
      if (costs[i+1][j+1] == ins):
        backptr[i+1][j+1] = EditOp.Insertion
      elif (costs[i+1][j+1] == del_):
        backptr[i+1][j+1] = EditOp.Deletion
      elif (s1[i] == s2[j]):
        backptr[i+1][j+1] = EditOp.Correct
      else:
        backptr[i+1][j+1] = EditOp.Substitution

  if print_debug_info:
    print("Mincost: ", costs[n1][n2])
  i = n1
  j = n2
  # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
  # is a position in n2.
  while (i > 0 or j > 0):
    if print_debug_info:
      print("i: ", i, " j: ", j)
    ed_op = EditOp.Correct
    if (i >= 0 and j >= 0):
      ed_op = backptr[i][j]
    if (i >= 0 and j < 0):
      ed_op = EditOp.Deletion
    if (i < 0 and j >= 0):
      ed_op = EditOp.Insertion
    if (i < 0 and j < 0):
      raise RuntimeError("Invalid alignment")
    
    if (ed_op == EditOp.Insertion):
      align.append((-1, j-1))
      j -= 1
    elif (ed_op == EditOp.Deletion):
      align.append((i-1, -1))
      i -= 1
    else:
      align.append((i-1, j-1))
      i -= 1
      j -= 1

  align.reverse()
  return costs[n1][n2], align
