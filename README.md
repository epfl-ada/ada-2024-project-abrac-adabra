# Mutation-driven insights into protein structure and ligand group prediction through protein and ligand embeddings
### A project by Abigaïl Ingster, Alexis Cogne, Riccardo Carpineto, Tancredi Cogne, and Viola Renne

## Abstract

## Git Structure
### Results
All results can be found in the notebooks P2 results and P3 results.
- P2 results contains data exploration, processing, and investigations of our first ideas
- P3 results contains the final results of our study which were elaborated on the basis on P2 results

### Data
Not all data files can be loaded on Git due to their size but the folder structure is still shown to easily reproduce the results. The folder Data contains subfolders and files:
- BindingDB: contains the file 'BindingDB_All.tsv' needed for data processing that can be found on the BindingDB website
- Processed Data: contains merged.csv and mutants.csv
- pdb_files: which contains the pdb files needed to represent the 3D structures of the proteins analyzed in our study (EGFR_HUMAN.pdb and KAPCA_BOVIN.pdb). These were retrieved from AlphaFold's online API.
- pair_dfs: all dataframes needed to visualize the IC50 scatter plots. There is one csv file per interaction pair (5).
- prot_viz: all dataframes needed to visualize the proteins. There is one csv file per interaction pair (5). 
- interactoin_pairs.csv: summarizes each selected interaction pair (that has at least 10 mutants). For each pair, the ligand name, the ligand SMILES, and the WT protein name are given. 
### Plots
All the plots that will be put on the website (either as html files or images) can be found in the folder Plots which contains subfolders and files:
- EGFR_HUMAN: html viualizations of the 3D structure of the WT EGFR_HUMAN and its mutants. This folder also contains mutant_names.csv which summarizes the name of the mutant associated to a given html file name in the current folder
- KAPCA_BOVIN: html viualizations of the 3D structure of the WT KAPCA_BOVIN and its mutants. This folder also contains mutant_names.csv which summarizes the name of the mutant associated to a given html file name in the current folder
- barplot_EGFR_HUMAN.png
- barplot_KAPCA_BOVIN.png
- interactive_ic50_plot.html
- ligands TODO
### Scripts
All scripts to process the data, generate visualization and results can be found in the src/scripts folder.
TODO
## Usage
### Data Processing
To process the data, run the file src/scripts/data_processing.py:
- it loads and processes the file BindingDB_All.tsv which is located in the folder TODO
- it generates two files both located in the folder TODO:
    - merged.csv: cleaned and processed data
    - mutants.csv: filtered merged.csv to retain valuable infos about mutants

### Generate Visualizations for the Website
To generate the visualization for the website, run the file TODO:
TODO