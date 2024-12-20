# Mutation-driven insights into protein structure and ligand group prediction through protein and ligand embeddings
### A project by Abigaïl Ingster, Alexis Cogne, Riccardo Carpineto, Tancredi Cogne, and Viola Renne
Link to our datastory: https://alexiscogne.github.io/ada-story-epfl/
## Introduction
Proteins and molecules interact all the time in living organisms. Experimentally analyzing known interactions and trying new ones has allowed a better understanding of biological processes in these organisms. These experiments are conducted in the lab and the results often appear in various scientific journals which makes it hard to find a large amount of results about a similar interaction. BindingDB is an online database that collected data from those experiments to allow an in silico analysis of the results. Our team was interested in digging into this database to see if valuable information could be obtained by an analysis of experiments made all over the world by many different generations of scientists. After exploring the database for a bit of time, we came up with four questions that we wanted to answer:

- Can we identify key mutation sites that alter binding affinity?
- How do mutations in reference-mutant protein groups affect binding affinities with specific ligands?
- Can we predict the coarse pattern of interaction of an unseen protein using its learned sequence embedding and ligand clustering?
- Can a low-dimensional space capture meaningful protein-ligand behaviors? E.g., protein embeddings cluster according to their similarity and clusters can be labeled according to their binding affinity with ligand groups.

Our analysis is separated in two parts: the first one, called “Mutants Analysis”, tackles the first two questions and the second one the two last.
## Results
All results can be found in the notebooks P2 results and P3 results.
- P2 results notebook contains data exploration, processing, and investigations of our first ideas
- P3 results notebook contains the final results of our study which were elaborated on the basis of P2 results

## Team's collaboration
- Tancredi: worked on data processing, mutant analysis, and data story
- Alexis: worked on data exploration, implemented the website, and data story
- Viola: worked on mutant analysis and data story
- Riccardo: worked on protein classification based on best-binding ligand class and datastory
- Abigaïl: worked on protein classification based on best-binding ligand class, graph analysis of protein and ligands, and datastory

## Data
Not all data files can be loaded on Git due to their size but the folder structure is still shown to easily reproduce the results. The folder Data contains subfolders and files:
- BindingDB: contains the file 'BindingDB_All.tsv' needed for data processing that can be found on the BindingDB website
- processed_data: contains merged.csv, mutants.csv, filtered_best_ligand_ic50.csv, unique_ligand_embeddings.csv
- pdb_files: which contains the pdb files needed to represent the 3D structures of the proteins analyzed in our study (EGFR_HUMAN.pdb and KAPCA_BOVIN.pdb). These were retrieved from AlphaFold's online API.
- pair_dfs: all dataframes needed to visualize the IC50 scatter plots. There is one csv file per interaction pair (5).
- prot_viz: all dataframes needed to visualize the proteins. There is one csv file per interaction pair (5). 
- interaction_pairs.csv: summarizes each selected interaction pair (that has at least 10 mutants). For each pair, the ligand name, the ligand SMILES, and the WT protein name are given. 

## Plots
All the plots that will be put on the website (either as html files or images) can be found in the folder Plots which contains subfolders and files:
- EGFR_HUMAN: html visualizations of the 3D structure of the WT EGFR_HUMAN and its mutants. This folder also contains mutant_names.csv which summarizes the name of the mutant associated to a given html file name in the current folder
- KAPCA_BOVIN: html visualizations of the 3D structure of the WT KAPCA_BOVIN and its mutants. This folder also contains mutant_names.csv which summarizes the name of the mutant associated to a given html file name in the current folder
- ligands_3D: html visualizations of the 3D structure of the 5 ligands.
- barplot_EGFR_HUMAN.png
- barplot_KAPCA_BOVIN.png
- interactive_ic50_plot.html

## Scripts
All scripts to process the data, generate visualization and results can be found in the src/scripts folder which contains:
- run website_plots_script.py to generate all visualizations for the website
- website_plots.py: contains all the methods needed to generate visualizations for the website
- mutant_analysis.py: contains all the methods needed for the mutant analysis 

## Usage

### Data Processing
To process the data, run the file src/scripts/data_processing.py:
- it loads and processes the file BindingDB_All.tsv which is located in the folder BindingDB
- it generates two files both located in the folder data/processed_data:
    - merged.csv: cleaned and processed data
    - mutants.csv: filtered merged.csv to retain valuable infos about mutants

### Mutants Analysis
The mutant analysis functions are located in src/scripts/mutants_analysis.py. This script includes functions for alignments, difference calculations, and IC50 plot generation.
Usage examples can be found in the accompanying Jupyter notebook (mutant_analysis.ipynb).
The primary inputs for these functions are derived from the data processing step.
