
# Deciphering Protein-Ligand Binding Dynamics: Analyzing Mutation Effects in BindingDB

## Abstract:
This project aims to investigate how mutations in protein sequences affect ligand binding affinities, focusing on groups of proteins (reference-mutant groups) with a shared ligand. By standardizing mutation annotations in the dataset, we will develop an analysis pipeline that compares each mutated sequence to a reference, identifying mutation positions and types. Graphs of IC50 values for these mutations will help reveal critical sites affecting binding, with visual indicators for amino acid property changes. Additionally, we aim to capture high-dimensional protein sequence features by using a deep-learning-based encoder for embedding these sequences into a lower-dimensional space. Dimensionality reduction will help us cluster proteins with similar binding behaviors, enabling us to build a predictive model for ligand affinity based on sequence embeddings. This research could provide valuable insights into how mutations influence binding, with applications in drug discovery and protein engineering.

### Research Questions:

1) How do mutations in reference-mutant protein groups affect binding affinities with specific ligands?
2) Can we identify key mutation sites that consistently alter binding affinity?
3) Is it possible to predict potential ligands for unseen proteins using learned sequence embeddings?
4) Can protein-ligand binding behaviors be clustered and visualized in a meaningful low-dimensional space?

### Proposed Additional Datasets:
No additional datasets are currently proposed. However, pre-trained models for sequence and ligand embedding are used.
In particular, for sequence embedding ESM2 is used.

### Methods:
1) Preprocessing: the following image shows the preprocessing steps done to perform analysis on mutant sequences.  
To focus on mutants, we created a new dataframe by grouping entries by ‘Ligand SMILES’ and ‘UniProt (SwissProt) Entry Name of Target Chain.’ Notably, while mutants of a wildtype share a ‘UniProt Entry Name,’ they differ in ‘Target Name’. We retained only groups with multiple entries (to include mutants) and saved lists of mutant names for each unique ‘Ligand SMILES’/‘UniProt Entry Name’ pair.
![Data Processing](images/ada_data_processing_pipeline.png)

2) Mutation Standardization and Analysis: Use sequence Needleman-Wunsch algorithm to identify and annotate mutations.
3) IC50 Graphing and Visualization: For each reference-mutant group and ligand, generate IC50 variation graphs with annotations for mutation type and amino acid property changes.
4) Embedding and Dimensionality Reduction: Use a deep-learning encoder to generate embeddings for sequences and ligands, followed by PCA or t-SNE to reduce dimensionality.
5) Clustering and Classification: Cluster proteins in the reduced space and apply a k-NN classifier to identify binding neighborhoods and predict ligands for new proteins.

### Proposed Timeline:
Week 0: Standardize mutation representation in dataset and conduct initial IC50 analysis.  [Already done]  
Week 1: Generate IC50 graphs for selected reference-mutant groups; explore amino acid property visualizations.  
Week 2: Test embedding models, apply dimensionality reduction, and visualize clusters.  
Week 3: Train and validate k-NN classifier on clustered embeddings.  
Week 4: Finalize analysis, visualize findings, and prepare project report.  

### Organization within the Team:
Milestone 1 (Mutation Standardization): Initial dataset processing and mutation annotation. [Already done]  
Milestone 2 (IC50 Graphing): Complete visualizations for key mutations across reference-mutant groups.  
Milestone 3 (Embedding and Clustering): Generate and reduce embeddings; build clustering model.  
Milestone 4: Final project wrap-up, report preparation, and presentation.  


```bash
# clone project
git clone git@github.com:epfl-ada/ada-2024-homework-1-abrac-adabra.git
cd ada-2024-homework-1-abrac-adabra.git

# install requirements
pip install -r pip_requirements.txt
```


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

