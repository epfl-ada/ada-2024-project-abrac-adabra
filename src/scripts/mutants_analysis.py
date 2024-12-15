from Bio import Align
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from src.models.ProteinModel import ProteinModel
import seaborn as sns
import re


amino_acid_dict = {
    'A': 'Alanine',
    'C': 'Cysteine',
    'D': 'Aspartic Acid',
    'E': 'Glutamic Acid',
    'F': 'Phenylalanine',
    'G': 'Glycine',
    'H': 'Histidine',
    'I': 'Isoleucine',
    'K': 'Lysine',
    'L': 'Leucine',
    'M': 'Methionine',
    'N': 'Asparagine',
    'P': 'Proline',
    'Q': 'Glutamine',
    'R': 'Arginine',
    'S': 'Serine',
    'T': 'Threonine',
    'V': 'Valine',
    'W': 'Tryptophan',
    'Y': 'Tyrosine'
}


def convert_aa_names(substitution):
    """
    Converting the amino acids in the string into the amino acid names.
    """
    if substitution != 'Deletion':
        splits = substitution.split(' -> ')
        if len(splits) > 1:
            aa1 = substitution.split(' -> ')[0]
            aa2 = substitution.split(' -> ')[1]
            return f'{amino_acid_dict[aa1]} -> {amino_acid_dict[aa2]}'
        else:
            return amino_acid_dict[substitution]
    return substitution


def find_insertion(name):
    """
    Finds the insertion amino acid in the name.
    :param name: The name of the mutant.
    :return: The position of the insertion and the inserted amino acids.
    """
    pattern = r".*\[\d+-(\d+),['’\\]+(\w+)['’\\]+,\d+-\d+\]"
    match = re.match(pattern, name)

    if match:
        return int(match.group(1)), match.group(2)


def compute_pairwise_alignment(sequence_1, sequence_2):
    """
    Compute the differences between two sequences by aligning the two sequences.
    :param sequence_1: first sequence.
    :param sequence_2: second sequence.
    :return: tuple with the two sequences alignments.
    """
    aligner = Align.PairwiseAligner(match_score=1.0, mode="global", mismatch_score=-2.0, gap_score=-3.5,
                                    query_left_extend_gap_score=0, query_internal_extend_gap_score=0,
                                    query_right_extend_gap_score=0, target_left_extend_gap_score=0,
                                    target_internal_extend_gap_score=0, target_right_extend_gap_score=0)

    alignments = aligner.align(sequence_1, sequence_2)
    # Get the best alignment
    align_array = np.array(alignments[0])
    return align_array[0], align_array[1]


def compute_multiple_alignment(reference_protein, mutants_list, sequences_list):
    # Compute pairwise alignments between reference and other sequences
    differences = compute_reference_mutant_differences(reference_protein, mutants_list, sequences_list)
    condition = np.all(differences['Insertion Positions'].apply(lambda x: x.size == 0) == False)

    # Initialize
    aligned_sequences = differences[['Mutant Name', 'Alignment Mutant No Insertion']].copy()
    aligned_sequences = aligned_sequences.rename(columns={'Mutant Name': 'Protein Name',
                                                          'Alignment Mutant No Insertion': 'Alignment'})
    new_row = pd.DataFrame({'Protein Name': [reference_protein],
                            'Alignment': [differences['Alignment Reference No Insertion'][0]]})
    aligned_sequences = pd.concat([new_row, aligned_sequences], ignore_index=True)

    if condition:
        return aligned_sequences

    # Create alignment
    positions = len(differences['Alignment Reference No Insertion'][0])
    i, j, c = 0, 0, 0
    while i < positions:
        value = [j in row['Insertion Positions'] for index, row in differences.iterrows()]
        condition = [j + 1 in row['Insertion Positions'] and value[index] for index, row in differences.iterrows()]
        if np.any(value):
            positions += 1
            for idx, row in aligned_sequences.iterrows():
                if idx == 0:
                    aligned_sequences.loc[idx, 'Alignment'] = (aligned_sequences.loc[idx, 'Alignment'][:i] + '-' +
                                                               aligned_sequences.loc[idx, 'Alignment'][i:])
                else:
                    index = idx - 1
                    if value[index]:
                        aligned_sequences.loc[idx, 'Alignment'] = (aligned_sequences.loc[idx, 'Alignment'][:i] +
                                                                     differences.loc[index, 'Alignment Mutant'][j] +
                                                                     aligned_sequences.loc[idx, 'Alignment'][i:])
                    else:
                        aligned_sequences.loc[idx, 'Alignment'] = (aligned_sequences.loc[idx, 'Alignment'][:i] + '-' +
                                                                     aligned_sequences.loc[idx, 'Alignment'][i:])
            if not np.any(condition):
                i += c + 1
                c = 0
            else:
                c += 1
        i += 1
        j += 1

    return aligned_sequences


def compute_reference_mutant_differences(reference_protein, mutants_list, sequences_list, check=True):
    """
    Compute the differences between reference and mutants' sequences.
    :param reference_protein: name of the reference protein.
    :param mutants_list: list of names of protein mutants.
    :param sequences_list: list of sequences.
    :param check: True for the five proteins we are considering, False otherwise. It performs a manual check on the position of the insertion.
    :return: DataFrame containing the mutant name, the alignment of the reference, the alignment of the mutant and the
    positions at which there is a difference in the alignments.
    """
    reference_index = mutants_list.index(reference_protein)
    reference_sequence = sequences_list[reference_index]
    # Remove all digits using re.sub
    reference_sequence = re.sub(r'\d', '', reference_sequence)

    comparison_results = []
    for i in range(len(mutants_list)):
        if mutants_list[i] == reference_protein:
            continue
        mutant = mutants_list[i]
        mutant_sequence = sequences_list[i]
        # Remove all digits using re.sub
        mutant_sequence = re.sub(r'\d', '', mutant_sequence)
        # Get the alignments
        alignment_1, alignment_2 = compute_pairwise_alignment(reference_sequence, mutant_sequence)

        # Check if there are insertions
        insertion_index = np.where(alignment_1 == b"-")[0]

        # Remove insertions
        alignment_1_no_insertion = np.delete(alignment_1, insertion_index)
        alignment_2_no_insertion = np.delete(alignment_2, insertion_index)

        if check and len(insertion_index) > 0:  # This is a manual check for just the 5 groups we look at
            position_1, insert = find_insertion(mutant)
            if insertion_index[0] != position_1:
                insertion_index = np.arange(position_1, position_1 + len(insert))


        # See where deletion or substitutions are considering the reference protein
        deletion_index = np.where(alignment_2_no_insertion == b"-")[0]
        substitution_index = np.where((alignment_2_no_insertion != b"-") & (alignment_2_no_insertion != alignment_1_no_insertion))[0]

        comparison_results.append({
            'Mutant Name': mutant,
            'Alignment Reference': ''.join([byte.decode('utf-8') for byte in alignment_1]),
            'Alignment Mutant': ''.join([byte.decode('utf-8') for byte in alignment_2]),
            'Alignment Reference No Insertion': ''.join([byte.decode('utf-8') for byte in alignment_1_no_insertion]),
            'Alignment Mutant No Insertion': ''.join([byte.decode('utf-8') for byte in alignment_2_no_insertion]),
            'Insertion Positions': insertion_index,
            'Deletion Positions': deletion_index,
            'Substitution Positions': substitution_index
        })

    results = pd.DataFrame(comparison_results)
    condition = ((results['Insertion Positions'].apply(lambda x: x.size == 0)) &
                 (results['Deletion Positions'].apply(lambda x: x.size == 0)) &
                 (results['Substitution Positions'].apply(lambda x: x.size == 0)))
    results_clean = results[~condition]
    return results_clean


def find_ic50(df_merged, proteins, ligand):
    """
    Find IC50 values associated with the given protein list and ligands.
    :param df_merged: DataFrame with BindingDB information.
    :param proteins: list of proteins names.
    :param ligand: ligand smile.
    :return: Series containing IC50 values for each protein in the list and the given ligand.
    """
    return df_merged[(df_merged['Ligand SMILES'] == ligand) &
                     (df_merged['Target Name'].isin(proteins))].set_index('Target Name')['IC50 (nM)']


def define_mutation(row):
    """
    Define for the given row, the type of mutation. If it is a substitution,
    the previous and current amino acids. If it is an insertion, which amino acid is inserted.
    :param row: row of a dataframe.
    :return: the exact substitution or insertion that occurred.
    """
    if row['Type'] == 'gap':
        return 'Deletion'
    elif row['Type'] == 'insertion':
        return f"{row['Alignment Mutant'][row['Positions']]}"
    else:
        return f"{row['Alignment Reference No Insertion'][row['Positions']]} -> {row['Alignment Mutant No Insertion'][row['Positions']]}"


def compute_probability_variation(row, model):
    """
    Compute the difference between probabilities given from ESM2.
    :param row: row of the dataframe with the mutation or gap.
    :param model: ESM2.
    """
    if row['Type'] == 'substitution':
        probabilities = pd.DataFrame(model.get_probabilities_unmasked(row['Alignment Reference'], row['Positions']))
        return probabilities[probabilities['token_str'] == row['Mutation'][5]]['score'].values[0] - probabilities[probabilities['token_str'] == row['Mutation'][0]]['score'].values[0]
    else:
        return 0


def compute_variation_ic50(row, df_merged):
    """
    Plot IC50 graph for a specific row of the mutants dataframe.
    :param row: selected row of the mutants dataframe to plot.
    :param df_merged: DataFrame containing the BindingDB information.
    :returns: DataFrame used for plotting.
    """
    # Find ic50 values associated with the mutants and the reference protein
    ic50_df = find_ic50(df_merged, row['Target Names'], row['Ligand SMILES'])

    if not ic50_df.index.is_unique:
        print('For this ligand-protein pair there are multiple values of IC50 and we decided to drop this case.')
        return None, None, None

    # Get position with differences
    differences = compute_reference_mutant_differences(row['WT Target Name'], row['Target Names'], row['BindingDB Target Chain Sequence'])

    # Set colours
    all_colors = list(mcolors.CSS4_COLORS.keys())
    differences['Colour'] = random.sample(all_colors, differences.shape[0])

    # Set IC50
    reference_ic50 = ic50_df.loc[row['WT Target Name']]
    differences['IC50 Difference'] = differences['Mutant Name'].map(ic50_df - reference_ic50)
    if reference_ic50 != 0:
        differences['IC50 Ratio'] = differences['Mutant Name'].map(ic50_df / reference_ic50)
        differences['IC50 Log Ratio'] = differences['Mutant Name'].map(np.log10(ic50_df / reference_ic50))
        differences['IC50 Percentage'] = differences['Mutant Name'].map((ic50_df - reference_ic50) / reference_ic50 * 100)

    # Combine the positions
    differences['Position'] = differences.apply(
        lambda row:
        [(pos, 'substitution') for pos in row['Substitution Positions']] +
        [(pos, 'gap') for pos in row['Deletion Positions']] +
        [(pos, 'insertion') for pos in row['Insertion Positions']], axis=1
    )

    differences_explode = differences.explode('Position').dropna()
    differences_explode[['Positions', 'Type']] = pd.DataFrame(differences_explode['Position'].tolist(), index=differences_explode.index)
    differences_explode.drop(columns=['Position', 'Insertion Positions', 'Substitution Positions', 'Deletion Positions'], inplace=True)
    differences.drop(columns=['Position'], inplace=True)
    differences_explode['Mutation'] = differences_explode.apply(define_mutation, axis=1)

    # Compute probabilities
    model = ProteinModel(model="facebook/esm2_t6_8M_UR50D")
    differences_explode['Probability Difference'] = differences_explode.apply(compute_probability_variation, axis=1, args=(model,))

    # Remove unused columns
    differences_explode.drop(['Alignment Reference', 'Alignment Mutant', 'Alignment Reference No Insertion', 'Alignment Mutant No Insertion'], axis=1, inplace=True)

    differences_explode['Mutation'] = differences_explode['Mutation'].apply(convert_aa_names)

    # Increase the group by when it is true. The formula is true when:
    # 1) It is a different mutant
    # 2) It is a substitution
    # 3) The position difference is greater than 1
    differences_explode["Group"] = (
        (differences_explode["Mutant Name"].ne(differences_explode["Mutant Name"].shift()) |
         ((differences_explode["Type"] == 'gap') & (differences_explode["Positions"].diff() != 1)) |
         (differences_explode["Type"] == 'substitution') |
         ((differences_explode["Type"] == 'insertion') & (differences_explode["Positions"].diff() != 1))
         ).cumsum()
    )

    columns = list(differences_explode.columns)
    columns.remove('Positions')
    columns.remove('Mutation')
    agg_funcs = {"Positions": list,
                 "Mutation": list}
    agg_funcs.update({column: 'first' for column in columns})
    grouped_df = differences_explode.groupby('Group', as_index=False).agg(agg_funcs)

    grouped_df['Mutation'] = grouped_df.apply(
        lambda row: row['Mutation'][0] if row['Type'] != 'insertion' else row['Mutation'],
        axis=1
    )

    return differences_explode, grouped_df, differences


def plot_ic50_graph(row, df_merged, ic50_column='IC50 Difference'):
    """
    Plot graph where the x-axis represents the amino acid sequence, and the y-axis represent the difference IC50 value
    :param row: row with information on the reference and mutants.
    :param df_merged: DataFrame containing the columns Positions, Colours and Type.
    :param ic50_column: column to use as the y-axis for the plot.
    """
    wt_name = row['WT Target Name']
    df, _, _ = compute_variation_ic50(row, df_merged)

    if df is None:
        return None

    plt.figure(figsize=(8, 6))
    df['Marker'] = df.Type.apply(lambda x: 'x' if x == 'substitution' else 'o')
    seen_mutants = set()
    for _, row in df.iterrows():
        mutant = row['Mutant Name'].replace(wt_name, '')
        if mutant not in seen_mutants:
            plt.scatter(row['Positions'], row[ic50_column], marker=row.Marker, s=100, color=row['Colour'], alpha=0.75, label=mutant)
            seen_mutants.add(mutant)
        else:
            plt.scatter(row['Positions'], row[ic50_column], marker=row.Marker, s=100, color=row['Colour'], alpha=0.75)

    plt.xlabel('Amino Acid Position')
    plt.ylabel('Variation in IC50 Value')
    plt.title(f'Variation in IC50 Values by Amino Acid Position for mutants of {wt_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ic50_graph_with_probabilities(row, df_merged, ic50_column='IC50 Difference'):
    """
    Plot graph where the x-axis represents the amino acid sequence, and the y-axis represent the difference IC50 value
    where the colours represent the ESM2 variations in probability
    :param row: row with information on the reference and mutants.
    :param df_merged: DataFrame containing the columns Positions, Colours and Type.
    :param ic50_column: column to use as the y-axis for the plot.
    """
    wt_name = row['WT Target Name']
    df, _, _ = compute_variation_ic50(row, df_merged)

    # Do not plot if the case was dropped
    if df is None:
        return None

    x_min = df["Positions"].min()
    x_max = df["Positions"].max()
    buffer = (x_max - x_min) * 0.05
    left_lim = x_min - buffer
    right_lim = x_max + buffer

    df.drop(df[df['Type'] == 'gap'].index, inplace=True)

    sns.set(style="whitegrid")

    g = sns.scatterplot(
        data=df,
        x="Positions",
        y=ic50_column,
        hue="Probability Difference",
        palette="viridis",
        style="Type",
        markers={'substitution': 'X'}
    )

    norm = plt.Normalize(df['Probability Difference'].min(), df['Probability Difference'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cbar = plt.colorbar(sm, ax=g)
    cbar.set_label("Difference in ESM2 Probability", fontsize=10)

    plt.legend().remove()
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Variation in IC50 Value')
    plt.title(f'Variation in IC50 Values by Amino Acid Position for mutants of {wt_name}', fontsize=10)
    plt.xlim(left_lim, right_lim)
    plt.tight_layout()
    plt.show()
