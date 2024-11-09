from Bio import Align
import pandas as pd
import numpy as np
import json


def get_protein_sequence(file_path, sequence_name):
    """
    Find the sequence of the given protein name from the given file path.
    :param file_path: path to the file containing the protein sequences.
    :param sequence_name: name of the protein.
    """
    df = pd.read_csv(file_path)
    return df[df['Target Name'] == sequence_name]['BindingDB Target Chain Sequence'].iloc[0]


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


def get_differences(proteins_list_str, file_path):
    """
    Compute the differences between two sequences by aligning the two sequences.
    :param proteins_list_str: list of protein sequences.
    :param file_path: path to the file containing the protein sequences.
    :return: DataFrame containing the mutant name, the alignment of the reference, the alignment of the mutant and the
    positions at which there is a difference in the alignments.
    """
    proteins_list = json.loads(proteins_list_str.replace("'", '"'))
    sequence1 = get_protein_sequence(file_path, proteins_list[0])

    comparison_results = []
    for protein in proteins_list[1:]:
        sequence2 = get_protein_sequence(file_path, protein)
        alignment, positions = compute_differences(sequence1, sequence2)
        comparison_results.append({
            'Protein Name': protein,
            'Alignment 1': ''.join([byte.decode('utf-8') for byte in alignment[0]]),
            'Alignment 2': ''.join([byte.decode('utf-8') for byte in alignment[1]]),
            'Positions': positions
        })

    return pd.DataFrame(comparison_results)
