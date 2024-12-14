import ast
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import nglview as nv
import py3Dmol
import plotly.graph_objects as go
import torch
from src.scripts.mutants_analysis import *
from plotly.subplots import make_subplots


def get_ligand_name_from_smiles(smiles):
    """
    Query ligand name from SMILES on pubchem
    :param smiles: Ligand SMILES 
    :return: ligand name from SMILES as defined on pubchem
    """
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/IUPACName/JSON".format(smiles)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0]['IUPACName']
    except Exception as e:
        print(f"Error retrieving name for SMILES {smiles}: {e}")
        return None
    
def save_dfs_ligands(df_merged, df_mutants):
    """
    Save infos about interaction pairs displayed on the website
    :param df_merged: dataframe containing processed data
    :param df_mutants: dataframe containing infos about mutants after data processing
    :return: interaction pair names of the structure [Ligand SMILES, Uniprot Name of WT]
    """
    saving_folder_dfs = "../data/pair_dfs"
    saving_folder_dfs_test = "../data/test"
    pair_numbers = 0
    ligand_smiles = []
    wt_names= []

    # Filtering pairs and saving dfs and ligand infos
    for _, row in df_mutants.iterrows():
        # Filter to keep only interaction pairs with at least 10 mutants
        if len(row['Target Names']) > 10:
            df_pair, test = compute_variation_ic50(row, df_merged)
            if df_pair is None:
                # See P2 first cell for explanation of this problem TODO
                print("This pair will not be saved due to multiple conflicting values in BindingDB") 
            else: 
                pair_numbers += 1
                ligand_smiles.append(row['Ligand SMILES'])
                wt_names.append(row['UniProt (SwissProt) Entry Name of Target Chain'])
                # Saving df to csv file for easier access later on
                saving_path_df = os.path.join(saving_folder_dfs, f'interaction_pair{pair_numbers}.csv')
                df_pair.to_csv(saving_path_df)
                saving_path_df_test = os.path.join(saving_folder_dfs_test, f'interaction_pair{pair_numbers}_test.csv')
                test.to_csv(saving_path_df_test)
                print("Pair information succesfully saved")
            print("---------------------------------------------------------------------------")

    ligand_names = [get_ligand_name_from_smiles(s) for s in ligand_smiles]
    return pd.DataFrame({'Ligand SMILES': ligand_smiles, 'Ligand name': ligand_names, 'WT protein': wt_names})

def visualize_ligands(all_smiles):
    """
    Save ligand 3D structure to display on the website
    :param all_smiles: smiles of each ligand that is present in one of the selected interaction pairs
    :return: all ligand names as defined on pubchem
    """

    saving_folder_ligands = "../plots/ligands"

    for idx, smiles in enumerate(all_smiles):
        # Creating and Saving Ligand Representation from SMILES
        mol = Chem.MolFromSmiles(smiles)

        # Generate 3D coordinates for the molecule
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Get 3D coordinates from the RDKit molecule
        conf = mol.GetConformer()
        coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

        viewer = py3Dmol.view(width=800, height=600)
        block = Chem.MolToMolBlock(mol)
        viewer.addModel(block, 'mol')

        viewer.setStyle({'stick': {}})
        viewer.setBackgroundColor('white')

        min_coords = [min(coord[i] for coord in coords) for i in range(3)]
        max_coords = [max(coord[i] for coord in coords) for i in range(3)]

        viewer.zoomTo({
            'center': {
                'x': (min_coords[0] + max_coords[0]) / 2,
                'y': (min_coords[1] + max_coords[1]) / 2,
                'z': (min_coords[2] + max_coords[2]) / 2,
            },
            'zoom': 0.5,  # Adjust zoom level if needed
        })

        # Saving ligand representation
        saving_path_ligand = os.path.join(saving_folder_ligands, f'ligand_{idx}')

        viewer.write_html(saving_path_ligand + '.html')

def generate_interactive_ic50_plot():
    """
    Generate IC50 interactive plot that will be displayed on the website
    """

    file_directory = '../data/pair_dfs'
    saving_directory = '../plots'

    buttons = []
    all_traces = []
    total_number_of_mutants_to_display = 0
    for file_name in os.listdir(file_directory):
        if file_name.endswith('.csv'):
            path = os.path.join(file_directory, file_name)
            df_pair = pd.read_csv(path)

            total_number_of_mutants_to_display += len(df_pair['Mutant Name'].unique())

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, x_title='Mutation Position', y_title='Log Ratio of IC50 values between WT and a given mutant')
    # Going over every saved interaction df
    for idx, file_name in enumerate(os.listdir(file_directory)):
        if file_name.endswith('.csv'):
            path = os.path.join(file_directory, file_name)

            df_pair = pd.read_csv(path)

            # Add traces for the current file
            visibility = []
            df_pair['Mutant Name'] = df_pair['Mutant Name'].str.replace('cAMP-dependent protein kinase catalytic subunit alpha', 'KAPCA_BOVIN') # TODO move to creation of df
            df_pair['Mutant Name'] = df_pair['Mutant Name'].str.replace('Epidermal growth factor receptor', 'EGFR') # TODO move to creation of df
            
            # Add every mutant to the plot
            for mutant_name in df_pair['Mutant Name'].unique():
                df_subset = df_pair[df_pair['Mutant Name'] == mutant_name]
                trace1 = go.Scatter(
                    x=df_subset['Positions'],
                    y=df_subset['IC50 Log Ratio'],
                    mode='markers',
                    name=mutant_name,
                    hovertemplate=(
                        "<b>Position:</b> %{x}<br>"
                        "<b>IC50 Log Ratio:</b> %{y}<br>"
                        "<b>Mutation:</b> %{customdata}<br>"
                    ),
                    text=df_subset['Mutant Name'],
                    customdata=df_subset['Mutation'],
                    
                    visible=False,
                    
                )
                df_subset_no_deletions = df_subset[df_subset.Type !='gap']
                trace2 = go.Scatter(
                    x=df_subset_no_deletions['Positions'],
                    y=df_subset_no_deletions['IC50 Log Ratio'],  # Replace with your actual parameter
                    mode='markers',
                    hovertemplate=(
                        "<b>Position:</b> %{x}<br>"
                        "<b>IC50 Log Ratio:</b> %{y}<br>"
                        "<b>Mutation:</b> %{customdata}<br>"
                    ),
                    name=mutant_name,
                    customdata=df_subset_no_deletions['Mutation'],
                    visible=False,
                    text=None,
                    marker=dict(
                        color=df_subset_no_deletions['Probability Difference'],  # Color by Other Parameter value
                        coloraxis='coloraxis'  # Choose a color scale to represent intensity
                    ),
                    showlegend=False
                )
                all_traces.append(trace1)
                all_traces.append(trace2)
                visibility.extend([True, True])

            start_index = len(all_traces)  - len(visibility)
            end_index = len(all_traces)
            file_visibility = [False] * total_number_of_mutants_to_display * 2
            for i in range(start_index, end_index):
                file_visibility[i] = True
                

            buttons.append({
                'label': f'Interaction pair {idx+1}',
                'method': 'update',
                'args': [
                    {'visible': file_visibility}
                ]
            })

            for trace in all_traces:
                trace.visible = False
            for i in range(start_index, end_index):
                all_traces[i].visible = True

    # Add all traces to the figure
    for idx, t in enumerate(all_traces):
        fig.append_trace(t, row=(idx%2)+1, col=1)
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'x': 0.7,
            'y': 1.105,
            'showactive': True,
            'active': 4
        }],
        title='Ratio of IC50 values between WT and Mutants by position',
        hovermode='closest'
    )

    fig.update_layout(coloraxis=dict(
            colorscale='RdBu',cmin=-1,cmax=1,  
            colorbar=dict(len=0.5, y=0.2)
        ))


    # Save the interactive plot
    fig.write_html(os.path.join(saving_directory, 'interactive_ic50_plot.html'))

# TBD
def viualize_mutant(protein_file, row=None):
    view = nv.show_file(protein_file)  # Update with the correct file path

    # Remove default representations
    if row.mutations is None:
        print("WT protein")
    else:
        print("Mutant: ", row.mutant_name)
        view.clear_representations()
        view.add_cartoon(color="#D3D3D3")

        for m in row.mutations:
            if m[0] == 'Deletion':
                view.add_cartoon(selection=m[1], color="blue")  # Highlight region 1-18 in blue
            else:
                view.add_ball_and_stick(selection=m[1], color="red")  # Highlight residue 858 in red
    nv.write_html(row.mutant_name + '.html', [view])

def generate_interactive_protein_structure_plot():
    # Load the PDB file
    test_df = pd.DataFrame({'mutant_name': ['WT', 'mutant [1-18]', 'mutant [C18->T]', 'mutant [1-18][C21->T]'], 'mutations': [None, ['Deletion', '1-18'], ['C18->T', '18'], [['Deletion', '1-18'], ['C21->T', '21']]]})

    for name, r in test_df.iterrows():
        viualize_mutant('pdb_files/P00533.pdb', r)

def convert_aa_names(string):
    if string!='Deletion':
        aa1 = string.split(' -> ')[0]
        aa2 = string.split(' -> ')[1]
        return f'{amino_acid_dict[aa1]} -> {amino_acid_dict[aa2]}'
    else:
        return 'Deletion'

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

"""
# Data loading
print("Started loading data")
df_mutants = pd.read_csv('../data/mutants.csv')
df_merged = pd.read_csv('../data/merged_df.csv')
df_mutants['Target Names'] = df_mutants['Target Names'].apply(lambda x: ast.literal_eval(x))
df_mutants['BindingDB Target Chain Sequence'] = df_mutants['BindingDB Target Chain Sequence'].apply(lambda x: ast.literal_eval(x))
print("Loaded data")


# Generate dfs and plots
interaction_pairs = save_dfs_ligands(df_merged, df_mutants)
interaction_pairs.to_csv('../data/interaction_pairs.csv')
print("Saved dfs")
visualize_ligands(interaction_pairs['Ligand SMILES'])
print("Saved ligand representations")
generate_interactive_ic50_plot()
print("Saved interactive IC50 plot")
# visualize mutants
"""
