
# Deciphering Protein-Ligand Binding Dynamics: Analyzing Mutation Effects in BindingDB

## Abstract:
This project aims to investigate how mutations in protein sequences affect ligand binding affinities, focusing on reference-mutant groups with a shared ligand. To identify mutation positions and types, we will use an alignment method that compares each mutated sequence to a reference. Potential key domains within the reference protein can be identified by plotting IC50 values against mutation positions and incorporating additional data on mutation characteristics, such as occurrence probability and chemical properties.
Additionally, we aim to develop a tool that assists researchers within the ligand space, replacing the usual resource-intensive, blind search with a protein-specific approach. In that aim, we will embed proteins into an euclidean space using a language model and cluster them into a low-dimensional space for visualization. Using their best-binding ligands as a label and after clustering ligands into coarse groups, we will train a classifier outputting a potential ligand group given a protein sequence.

### Research Questions:

1) How do mutations in reference-mutant protein groups affect binding affinities with specific ligands?  
2) Can we identify key mutation sites that alter binding affinity?  
3) Can we predict the coarse pattern of interaction of an unseen protein using its learned sequence embedding and ligand clustering?  
4) Can a low-dimensional space capture meaningful protein-ligand behaviors? E.g. protein embeddings cluster according to their similarity and clusters can be labeled according to their binding affinity with ligand groups.  

### Proposed Additional Datasets:
No additional datasets are currently proposed. However, pre-trained models for sequence and ligand embedding are used. In particular, for sequence embedding and ligand embeddings we use ESM2 and ChemBERTA respectively.

### Methods:
1) **Preprocessing:** the following image shows the preprocessing steps done to perform analysis on mutant sequences. To focus on mutants, we created a new dataframe by grouping entries by ‘Ligand SMILES’ and ‘UniProt (SwissProt) Entry Name of Target Chain’. We retained only groups with multiple entries (to include mutants) and saved lists of mutant names and sequences for each unique ‘Ligand SMILES’/‘UniProt Entry Name’ pair. [Already done]
![Data Processing](images/ada_data_processing_pipeline.png)

2) **Mutation Standardization and Analysis:** since the dataset does not provide a standard mutation format, we use the Needleman-Wunsch algorithm to automatically identify the differences between the reference sequence and the mutant ones. [Already done]  
3) **IC50 Graphing and Visualization:** for each reference-mutant group and ligand, report the IC50 variation between the reference protein and the mutants, taking into account both the mutation type and changes in amino acid properties. To characterize the nature of the mutations, we propose using an ESM2 masking model, where output probabilities are analyzed at the masked positions. We will need to account also the fact that most mutants have multiple mutations occurring at different positions.  
4) 
5) 

### Proposed Timeline:
**Week 9:** Standardize mutation representation in dataset and conduct initial IC50 analysis. [Already done]  
**Week 10:** Homework 2  
**Week 11:** Generate IC50 graphs for selected reference-mutant groups. Test embedding models, apply dimensionality reduction, and visualize clusters.  
**Week 12:** Explore amino acid properties for IC50 graphs part 1.   
**Week 13:** Explore amino acid properties for IC50 graphs part 2. Train and validate k-NN classifier on clustered embeddings.  
**Week 14:** Finalize analysis, visualize findings, and prepare project report.  

### Organization within the Team:
- **Team A:** (Viola, Tancredi & Alexis)
   - W11: Generate IC50 graphs for selected reference-mutant groups.  
   - W12: Explore amino acid properties for IC50 graphs using hydrophobic values.  
   - W13: Explore amino acid properties for IC50 graphs using ESM2.  
   - W14: Finalize analysis, visualize findings, and prepare project report.  
- **Team B:** (Riccardo & Abigail)  
   - W11: Test embedding models, apply dimensionality reduction, and visualize clusters.  
   - W12: 
   - W13:
   - W14: Finalize analysis, visualize findings, and prepare project report.  



```bash
# clone project
git clone git@github.com:epfl-ada/ada-2024-homework-1-abrac-adabra.git
cd ada-2024-homework-1-abrac-adabra.git

# install requirements
pip install -r pip_requirements.txt
```

