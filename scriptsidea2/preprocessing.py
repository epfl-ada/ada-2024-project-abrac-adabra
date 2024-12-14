# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch

def ligand_umap(unique_smiles_df, embeddings, dimensions=3, plotting=True, seed=42):
    # Normalize the features
    X_normalized = StandardScaler().fit_transform(embeddings)

    # Apply UMAP
    umap_model = umap.UMAP(n_components=dimensions, random_state=seed)
    X_transformed_UMAP = umap_model.fit_transform(X_normalized)

    # Create dataframe for UMAP values
    umap_df = pd.DataFrame(
        X_transformed_UMAP,
        columns=[f"Ligand UMAP {i}" for i in range(1, dimensions + 1)],
    )

    # Merge it with the SMILES dataframe
    merged_umap_df = pd.concat([unique_smiles_df, umap_df], axis=1)

    if plotting:
        sns.scatterplot(
            data=merged_umap_df, x="Ligand UMAP 1", y="Ligand UMAP 2", legend=False
        )
        plt.title("UMAP of ChemBERTa ligand embeddings")
        plt.show()

    return merged_umap_df


def cluster_ligand(merged_umap_df, n_clusters=9, plotting=True):

    kmeans_labels_UMAP = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
        merged_umap_df.iloc[:, 1:4]
    )
    merged_umap_df = pd.concat(
        [merged_umap_df, pd.DataFrame({"Ligand class": kmeans_labels_UMAP})], axis=1
    )

    if plotting:
        palette = sns.color_palette("tab10", n_colors=n_clusters)
        sns.scatterplot(
            data=merged_umap_df,
            x="Ligand UMAP 1",
            y="Ligand UMAP 2",
            hue="Ligand class",
            palette=palette,
            legend=False,
        )
        plt.title("k-means clustering of UMAPed ChemBERTa representations")
        plt.show()

    return merged_umap_df


def prot_umap(
    filtered_best_ligand_ic50, merged_umap_df, dimensions=3, n_clusters=9, plotting=True
):

    filtered_best_ligand_ic50.rename(
        columns={"Ligand SMILES_x": "Ligand SMILES"}, inplace=True
    )
    protein_ligand_matched = pd.merge(
        filtered_best_ligand_ic50, merged_umap_df, on="Ligand SMILES", how="inner"
    )
    # Convert strings to torch tensors
    protein_ligand_matched["ESM2 Embedding"] = protein_ligand_matched["ESM2 Embedding"].apply(lambda x: eval("torch." + x))

    X_prot = np.array([np.squeeze(x) for x in protein_ligand_matched["ESM2 Embedding"]])
    X_prot_normalized = StandardScaler().fit_transform(X_prot)

    umap_model = umap.UMAP(n_components=dimensions, random_state=17)
    X_prot_transformed_UMAP = umap_model.fit_transform(X_prot_normalized)

    umap_prot_df = pd.DataFrame(
        X_prot_transformed_UMAP,
        columns=[f"Protein UMAP {i}" for i in range(1, dimensions + 1)],
    )
    umap_prot_df.set_index(protein_ligand_matched.index, inplace=True)
    protein_ligand_matched = pd.concat([protein_ligand_matched, umap_prot_df], axis=1)

    if plotting:
        palette = sns.color_palette("tab10", n_colors=n_clusters)
        sns.scatterplot(
            data=protein_ligand_matched,
            x="Protein UMAP 1",
            y="Protein UMAP 2",
            hue="Ligand class",
            legend=False,
            palette=palette,
        )
        plt.title("Clustering of ESM2 protein embeddings, colored by ligand group")

    return protein_ligand_matched
