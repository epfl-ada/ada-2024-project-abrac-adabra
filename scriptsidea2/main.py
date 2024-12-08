from data_loading import load_data
from preprocessing import ligand_umap, cluster_ligand, prot_umap
from model_training import train_knn, train_svm, train_logistic, train_mlp
import numpy as np
import pandas as pd
import torch
def main():
    # Load data
    filtered_best_ligand_ic50, unique_smiles_df, embeddings = load_data(idea='idea2')
    
    # Preprocess data
    n_clusters = 9
    merged_umap_df = ligand_umap(unique_smiles_df, embeddings, dimensions = 3,plotting = False)
    merged_umap_df = cluster_ligand(merged_umap_df, n_clusters = n_clusters)
    protein_ligand_matched = prot_umap(filtered_best_ligand_ic50, merged_umap_df, dimensions = 3, n_clusters = n_clusters)

    X_prot = np.array([np.squeeze(x) for x in protein_ligand_matched['ESM2 Embedding']])
    y = protein_ligand_matched['Ligand class']

    # Train models
    knn_gscv = train_knn(X_prot, y);
    svm_clf = train_svm(X_prot, y);
    ll_gscv = train_logistic(X_prot, y);
    mlp_clf = train_mlp(X_prot, y,num_layers = 2);

if __name__ == "__main__":
    main()
