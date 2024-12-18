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
from src.scripts.website_plots import *
from src.scripts.mutants_analysis import bar_plot_df

# Data loading
print("Started loading data")
df_mutants = pd.read_csv('../../data/mutants.csv')
df_merged = pd.read_csv('../../data/merged_df.csv')
df_mutants['Target Names'] = df_mutants['Target Names'].apply(lambda x: ast.literal_eval(x))
df_mutants['BindingDB Target Chain Sequence'] = df_mutants['BindingDB Target Chain Sequence'].apply(lambda x: ast.literal_eval(x))
print("Loaded data")

# Generate dfs and plots
interaction_pairs = save_dfs_ligands(df_merged, df_mutants)
interaction_pairs.to_csv('../../data/interaction_pairs.csv')
print("Saved dfs")

generate_interactive_ic50_plot()
print("Saved interactive IC50 plot")

visualize_ligands(interaction_pairs['Ligand SMILES'])
print("Saved ligand representations")

mutants_infos = visualize_mutants('../../data/interaction_pairs.csv', '../../data/prot_viz', '../../data/pdb_files')
for k, v in mutants_infos.items():
    pd.DataFrame(v[1]).to_csv(f'../../plots/{k}/mutant_names.csv')

final_df = bar_plot_df('../../data/interaction_pairs.csv', '../../data/prot_viz')
custom_palette = sns.color_palette("rocket", as_cmap=True)(0.2, 0.8)
for name, group in final_df.groupby('WT protein'):
        plt.close()
        plt.figure(figsize=(10, 6), dpi = 300) 
        sns.set_style("darkgrid")
        custom_palette = sns.color_palette("rocket", n_colors=len(group['Ligand number'].unique())+1)
        custom_palette = custom_palette[1:] 
        sns.barplot(x='Mutant Name', y='IC50 Log Ratio', hue='Ligand number', data=group, errorbar=None, palette=custom_palette)
        plt.title(f'IC50 Log Ratio between WT {name} and mutants')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'../../plots/barplot_{name}.png', bbox_inches='tight');

print("Saved interactive mutant visualizations")