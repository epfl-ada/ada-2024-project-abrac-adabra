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
from Bio.PDB import PDBParser
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
    if not os.path.exists(saving_folder_dfs):
            os.makedirs(saving_folder_dfs)
    saving_folder_dfs_prot_viz = "../data/prot_viz"
    if not os.path.exists(saving_folder_dfs_prot_viz):
            os.makedirs(saving_folder_dfs_prot_viz)
    pair_numbers = 0
    ligand_smiles = []
    wt_names= []

    # Filtering pairs and saving dfs and ligand infos
    for _, row in df_mutants.iterrows():
        # Filter to keep only interaction pairs with at least 10 mutants
        if len(row['Target Names']) > 10:
            df_pair, _, df_protein_viz = compute_variation_ic50(row, df_merged)
            if df_pair is None:
                # See P2 first cell for explanation of this problem TODO
                print("This pair will not be saved due to multiple conflicting values in BindingDB") 
            else: 
                pair_numbers += 1
                ligand_smiles.append(row['Ligand SMILES'])
                wt_names.append(row['UniProt (SwissProt) Entry Name of Target Chain'])
                # Saving df to csv file for easier access later on
                df_pair['Mutant Name'] = df_pair['Mutant Name'].str.replace('cAMP-dependent protein kinase catalytic subunit alpha', 'KAPCA_BOVIN') 
                df_pair['Mutant Name'] = df_pair['Mutant Name'].str.replace('Epidermal growth factor receptor', 'EGFR_HUMAN')
                saving_path_df = os.path.join(saving_folder_dfs, f'interaction_pair{pair_numbers}.csv')
                df_pair.to_csv(saving_path_df)

                df_protein_viz['Mutant Name'] = df_protein_viz['Mutant Name'].str.replace('cAMP-dependent protein kinase catalytic subunit alpha', 'KAPCA_BOVIN') 
                df_protein_viz['Mutant Name'] = df_protein_viz['Mutant Name'].str.replace('Epidermal growth factor receptor', 'EGFR_HUMAN')
                saving_path_df_prot_viz = os.path.join(saving_folder_dfs_prot_viz, f'interaction_pair{pair_numbers}_prot_viz.csv')
                df_protein_viz.to_csv(saving_path_df_prot_viz)
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
    if not os.path.exists(saving_folder_ligands):
            os.makedirs(saving_folder_ligands)

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
        saving_path_ligand = os.path.join(saving_folder_ligands, f'ligand_{idx+1}')

        viewer.write_html(saving_path_ligand + '.html')

def generate_interactive_ic50_plot():
    """
    Generate IC50 interactive plot that will be displayed on the website
    """

    file_directory = '../data/pair_dfs'
    saving_directory = '../plots'
    if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

    buttons = []
    all_traces = []
    total_number_of_mutants_to_display = 0
    for file_name in sorted(os.listdir(file_directory)):
        if file_name.endswith('.csv'):
            path = os.path.join(file_directory, file_name)
            df_pair = pd.read_csv(path)

            total_number_of_mutants_to_display += len(df_pair['Mutant Name'].unique())

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, x_title='Mutation Position', y_title='Log Ratio of IC50 values between WT and a given mutant')
    # Going over every saved interaction df
    for idx, file_name in enumerate(sorted(os.listdir(file_directory))):
        if file_name.endswith('.csv'):
            path = os.path.join(file_directory, file_name)

            df_pair = pd.read_csv(path)

            # Add traces for the current file
            visibility = []
    
            # Add every mutant to the plot
            for mutant_name in df_pair['Mutant Name'].unique():
                df_subset = df_pair[df_pair['Mutant Name'] == mutant_name]
                trace1 = go.Scatter(
                    x=df_subset['Positions'],
                    y=df_subset['IC50 Log Ratio'],
                    mode='markers',
                    name=mutant_name,
                    marker_color=df_subset['Colour'],
                    hovertemplate=(
                        "<b>Position:</b> %{x}<br>"
                        "<b>IC50 Log Ratio:</b> %{y}<br>"
                        "<b>Mutation:</b> %{customdata}<br>"
                    ),
                    text=df_subset['Mutant Name'],
                    customdata=df_subset['Mutation'],
                    visible=False,
                )
                df_subset_substitutions = df_subset[df_subset.Type =='substitution']
                trace2 = go.Scatter(
                    x=df_subset_substitutions['Positions'],
                    y=df_subset_substitutions['IC50 Log Ratio'],  # Replace with your actual parameter
                    mode='markers',
                    hovertemplate=(
                        "<b>Position:</b> %{x}<br>"
                        "<b>IC50 Log Ratio:</b> %{y}<br>"
                        "<b>Mutation:</b> %{customdata}<br>"
                    ),
                    name=mutant_name,
                    customdata=df_subset_substitutions['Mutation'],
                    visible=False,
                    text=None,
                    marker=dict(
                        color=df_subset_substitutions['Probability Difference'],  # Color by Other Parameter value
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
            'x': 0.8,
            'y': 1.115,
            'showactive': True,
            'active': 4
        }],
        title='Ratio of IC50 values between WT and Mutants by position',
        hovermode='closest'
    )

    fig.update_layout(coloraxis=dict(
            colorscale='RdBu',cmin=-1,cmax=1,  
            colorbar=dict(len=0.35, y=0.2)
        ))


    # Save the interactive plot
    fig.write_html(os.path.join(saving_directory, 'interactive_ic50_plot.html'))

def new_visualize_mutants(interaction_pairs_path, df_folder, pdb_files_folder):
    """
    Save mutants 3D structure to display on the website
    :param interaction_pairs_path: path to df containing information about interaction pairs
    :param df_folder: folder containing csv files with infos to plot mutants
    :param pdb_files_folder: folder containing pdb files for WT proteins
    :return: dict where keys are WT protein names and values are (mutant_number, [mutant_names]) 
    """
    parser = PDBParser(QUIET=True)
    mutants_infos = {}
    interaction_pairs = pd.read_csv(interaction_pairs_path)
    for idx, file_name in enumerate(sorted(os.listdir(df_folder))):
        df = pd.read_csv(os.path.join(df_folder, file_name))
        wt_protein_name = interaction_pairs.iloc[idx]['WT protein']
        structure = parser.get_structure("protein", os.path.join(pdb_files_folder, f'{wt_protein_name}.pdb'))
        chain = list(structure.get_chains())[0]  # Assume the first chain

        view = nv.show_file(os.path.join(pdb_files_folder, f'{wt_protein_name}.pdb')) # Show WT protein
        curr_num_mutants, curr_mutant_names = mutants_infos[wt_protein_name] if wt_protein_name in mutants_infos.keys() else (0, [])

        saving_folder = f"../plots/{wt_protein_name}"
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        saving_path = os.path.join(saving_folder, f'{wt_protein_name}_mutant_0_viz.html') # Save WT protein viz
        nv.write_html(saving_path, [view])
        # Clean infos about mutation positions 
        for c in ['Insertion Positions', 'Deletion Positions', 'Substitution Positions']:
            df[c] = df[c].apply(lambda x: [int(i) for i in x.strip('[]').split(' ') if i != '' ])

        for _, row in df.iterrows():
            # Some mutants appear in multiple files and don't need to be represented multiple times
            if row['Mutant Name'] in curr_mutant_names:
                print("Mutant viz already saved")
            else:
                curr_num_mutants+=1
                view.clear()
                view.clear_representations()
                view = nv.show_file(os.path.join(pdb_files_folder, f'{wt_protein_name}.pdb')) # Show WT protein 
                view.clear_representations()
                view.add_cartoon(color="#D3D3D3")
                
                # One color per type of mutation
                insertions = row['Insertion Positions']
                deletions = row['Deletion Positions']
                substitutions = row['Substitution Positions']

                all = [a for a in range(len(chain)) if a not in deletions]
                start_range_nums = [all[0]+1]
                end_range_nums = []
                for i in range(len(all)-1):
                    if all[i]+1!=all[i+1]:
                        end_range_nums.append(all[i]+1)
                        start_range_nums.append(all[i+1]+1)
                end_range_nums.append(all[-1]+1)
                ranges = []

                for i in range(len(start_range_nums)):
                    curr_range = f'{start_range_nums[i]}-{end_range_nums[i]}'
                    ranges.append(curr_range)
                    view.add_cartoon(selection=curr_range, color=row['Colour'])
                
                for i in insertions:
                    view.add_spacefill(selection=f'{i+1}', color="blue", radius=1.5)
                for s in substitutions:
                    view.add_spacefill(selection=f'{s+1}', color="green", radius=1.5)

                saving_path = os.path.join(saving_folder, f'{wt_protein_name}_mutant_{curr_num_mutants}_viz.html')
                with open(saving_path, 'w') as f:
                    pass  # Do nothing, effectively clearing the file
                nv.write_html(saving_path, [view]) # Save mutant representation
                curr_mutant_names.append(row['Mutant Name']) 
        mutants_infos.update({wt_protein_name: (curr_num_mutants, curr_mutant_names)}) # Update summary table 

    return mutants_infos

def bar_plot_df(interaction_pairs_path, df_folder):
    """
    Create bar plot df to visualize difference in binding affinity based on the ligand for a given pair
    :param interaction_pairs_path: path to df containing information about interaction pairs
    :param df_folder: folder containing csv files with infos about binding affinity of mutants
    """
    interaction_pairs = pd.read_csv(interaction_pairs_path)
    mega_df = []
    for idx, file_name in enumerate(sorted(os.listdir(df_folder))):
        df = pd.read_csv(os.path.join(df_folder, file_name))
        df = df[['Mutant Name', 'IC50 Log Ratio']]
        df['WT protein'] = interaction_pairs.iloc[idx]['WT protein']
        df['Ligand SMILES'] = interaction_pairs.iloc[idx]['Ligand SMILES']
        df['Ligand name'] = interaction_pairs.iloc[idx]['Ligand name']
        df['Ligand number'] = idx+1
        mega_df.append(df)

    df_grouped = pd.concat(mega_df, ignore_index=True).groupby(['WT protein', 'Mutant Name'])[['Ligand number', 'IC50 Log Ratio']].agg(list).reset_index()
    df_grouped['Mutant Name'] = df_grouped['Mutant Name'].apply(lambda x: x.split(' ')[1])
    final_df = df_grouped[df_grouped['Ligand number'].apply(len) >=2].explode(['Ligand number', 'IC50 Log Ratio'])
    
    return final_df