from Bio import Align
import pandas as pd
import numpy as np
import json


def compute_differences(sequence_1, sequence_2):
    """
    Compute the differences between two sequences by aligning the two sequences.
    :param sequence_1: first sequence.
    :param sequence_2: second sequence.
    :return: tuple with the two sequences in the first position and the positions of the differences in the second.
    """
    aligner = Align.PairwiseAligner(match_score=1.0, mode="global", mismatch_score=-2.0, gap_score=-2.5,
                                    query_left_extend_gap_score=0, query_internal_extend_gap_score=0,
                                    query_right_extend_gap_score=0, target_left_extend_gap_score=0,
                                    target_internal_extend_gap_score=0, target_right_extend_gap_score=0)

    alignments = aligner.align(sequence_1, sequence_2)
    align_array = np.array(alignments[0])
    value = align_array[0] == align_array[1]
    positions = np.where(value == False)
    return align_array, positions[0]


def get_differences(reference_protein, mutants_list_str, sequences_list_str):
    """
    Compute the differences between two sequences by aligning the two sequences.
    :param reference_protein: name of the reference protein.
    :param mutants_list_str: list of names of protein mutants.
    :param sequences_list_str: list of sequences.
    :return: DataFrame containing the mutant name, the alignment of the reference, the alignment of the mutant and the
    positions at which there is a difference in the alignments.
    """
    mutants_list = json.loads(mutants_list_str.replace("'", '"'))
    sequences_list = json.loads(sequences_list_str.replace("'", '"'))
    reference_index = mutants_list.index(reference_protein)
    reference_sequence = sequences_list[reference_index]

    comparison_results = []
    for i in range(len(mutants_list)):
        if mutants_list[i] == reference_protein:
            continue
        mutant = mutants_list[i]
        mutant_sequence = sequences_list[i]
        alignment, positions = compute_differences(reference_sequence, mutant_sequence)
        comparison_results.append({
            'Mutant Name': mutant,
            'Alignment Reference': ''.join([byte.decode('utf-8') for byte in alignment[0]]),
            'Alignment Mutant': ''.join([byte.decode('utf-8') for byte in alignment[1]]),
            'Positions': positions
        })

    return pd.DataFrame(comparison_results)
