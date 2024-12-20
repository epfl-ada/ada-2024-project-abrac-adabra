import warnings
import os

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_data(interactions_df, filepath):
    """
    Load and return unique ligand SMILES and ligand embeddings 

    :param interactions_df: DataFrame containing the best protein-ligand interactions.
    :return: Tuple containing:
             - unique_smiles_df: DataFrame with unique ligand SMILES.
             - embeddings: DataFrame with ligand embeddings (loaded or computed).
    """
    # Extract unique ligand SMILES from interaction data
    unique_smiles_df = pd.DataFrame(
        {"Ligand SMILES": interactions_df["Ligand SMILES"].unique()}
    )
    
    embeddings = pd.read_csv(filepath)
    
    return unique_smiles_df, embeddings


def ligand_umap(unique_smiles_df, embeddings, dimensions=3, plotting=True, seed=42):
    """
    Apply UMAP on ligand embeddings and return UMAP-transformed data.

    :param unique_smiles_df: DataFrame with unique ligand SMILES.
    :param embeddings: DataFrame with ligand embeddings corresponding to the unique SMILES.
    :param dimensions: Number of dimensions for UMAP transformation (default is 3).
    :param plotting: Boolean to control if the UMAP plot should be displayed (default is True).
    :param seed: Random seed for reproducibility (default is 42).
    :return: DataFrame with UMAP-transformed ligand embeddings.
    """
    # Normalize the features
    X_normalized = StandardScaler().fit_transform(embeddings)

    # Apply UMAP transformation
    umap_model = umap.UMAP(n_components=dimensions, random_state=seed)
    X_transformed_UMAP = umap_model.fit_transform(X_normalized)
    umap_df = pd.DataFrame(
        X_transformed_UMAP,
        columns=[f"Ligand UMAP {i}" for i in range(1, dimensions + 1)],
    )

    # Merge UMAP results with SMILES data
    merged_umap_df = pd.concat([unique_smiles_df, umap_df], axis=1)

    # Plotting UMAP if required
    if plotting:
        sns.scatterplot(
            data=merged_umap_df, x="Ligand UMAP 1", y="Ligand UMAP 2", legend=False
        )
        plt.title("UMAP of ChemBERTa ligand embeddings")
        plt.show()

    return merged_umap_df


def cluster_ligand(merged_umap_df, n_clusters=9, plotting=True, save_path='data/processed_data/'):
    """
    Cluster ligands using k-means on UMAP-transformed embeddings and plot the results.

    :param merged_umap_df: DataFrame with UMAP-transformed ligand embeddings.
    :param n_clusters: Number of clusters for k-means (default is 9).
    :param plotting: Boolean to control if the clustering plot should be displayed (default is True).
    :param save_path: Path to save the generated plot.
    :return: DataFrame with ligand clusters added.
    """
    # Perform k-means clustering on UMAP-transformed data
    kmeans_labels_UMAP = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
        merged_umap_df.iloc[:, 1:4]
    )
    merged_umap_df = pd.concat(
        [merged_umap_df, pd.DataFrame({"Ligand class": kmeans_labels_UMAP})], axis=1
    )

    if save_path:
        merged_umap_df.to_csv(os.path.join(save_path, 'merged_umap_df.csv'))

    # Plotting clustering results
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
    interactions_df, ligand_umap_df, dimensions=3, n_clusters=9, plotting=True, save_path='data/processed_data/'
):
    """
    Apply UMAP on protein embeddings and merge them with ligand data for visualization.

    :param interactions_df: DataFrame with ligand-IC50 mapping.
    :param ligand_umap_df: DataFrame with UMAP-transformed ligand embeddings.
    :param dimensions: Number of dimensions for UMAP transformation (default is 3).
    :param n_clusters: Number of clusters for visualization (default is 9).
    :param plotting: Boolean to control if the UMAP plot should be displayed (default is True).
    :param save_path: Path to save the generated plot.
    :return: DataFrame with protein embeddings and ligand data.
    """

    protein_ligand_matched = pd.merge(
        interactions_df, ligand_umap_df, on="Ligand SMILES", how="inner"
    )
    # Convert protein embeddings to tensors
    protein_ligand_matched["ESM2 Embedding"] = protein_ligand_matched["ESM2 Embedding"].apply(lambda x: eval("torch." + x))

    # Normalize protein embeddings and apply UMAP
    X_prot = np.array([np.squeeze(x) for x in protein_ligand_matched["ESM2 Embedding"]])
    X_prot_normalized = StandardScaler().fit_transform(X_prot)

    umap_model = umap.UMAP(n_components=dimensions, random_state=17)
    X_prot_transformed_UMAP = umap_model.fit_transform(X_prot_normalized)

    # Create DataFrame with protein UMAP data
    umap_prot_df = pd.DataFrame(
        X_prot_transformed_UMAP,
        columns=[f"Protein UMAP {i}" for i in range(1, dimensions + 1)],
    )
    umap_prot_df.set_index(protein_ligand_matched.index, inplace=True)
    protein_ligand_matched = pd.concat([protein_ligand_matched, umap_prot_df], axis=1)

    if save_path:
        protein_ligand_matched.to_csv(os.path.join(save_path, 'protein_ligand_matched.csv'))

    # Plotting if required
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


def train_knn(X: np.ndarray, y: np.ndarray, cv=5):
    """
    Train a k-Nearest Neighbors classifier and perform grid search for optimal parameters.

    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds (default is 5).
    :return: Trained k-NN classifier with best parameters.
    """
    print("Training kNN...")
    knn_clf = KNeighborsClassifier()
    param_grid = {"n_neighbors": np.arange(1, 100)}
    knn_gscv = GridSearchCV(knn_clf, param_grid, cv=cv, scoring="balanced_accuracy")
    knn_gscv.fit(X, y)
    print(
        f"k-NN: best balanced accuracy of {knn_gscv.best_score_:.4f}, for {knn_gscv.best_params_}."
    )
    return knn_gscv


def train_svm(X: np.ndarray, y: np.ndarray, cv=5):
    """
    Train a Support Vector Machine (SVM) classifier with cross-validation.

    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds (default is 5).
    :return: Trained SVM classifier.
    """
    print("Training SVM...")
    svm_clf = SVC(random_state=17)
    score = cross_val_score(svm_clf, X, y, cv=cv, scoring="balanced_accuracy").mean()
    score_nb = cross_val_score(svm_clf, X, y, cv=cv, scoring="accuracy").mean()
    print(f"SVM: CV nb accuracy of {score:.4f}.")
    print(f"SVM: CV balanced accuracy of {score:.4f}.")
    return svm_clf


def train_logistic(X: np.ndarray, y: np.ndarray, cv=5):
    """
    Train a Logistic Regression model with grid search for optimal parameters.

    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds (default is 5).
    :return: Trained Logistic Regression classifier with best parameters.
    """
    print("Training Logistic...")
    # Suppress specific warnings
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

    ll_clf = LogisticRegression(max_iter=500, random_state=17)
    param_grid = {"multi_class": ["multinomial", "ovr"]}
    ll_gscv = GridSearchCV(ll_clf, param_grid, cv=cv, scoring="balanced_accuracy")
    ll_gscv.fit(X, y)

    print(
        f"Logistic Regression: best balanced accuracy of {ll_gscv.best_score_:.4f}, for the loss: {ll_gscv.best_params_}."
    )
    return ll_gscv


def train_mlp(X: np.ndarray, y: np.ndarray, cv=5, num_layers=5):
    """
    Train a Multi-Layer Perceptron (MLP) classifier with grid search for optimal parameters.

    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds (default is 5).
    :param num_layers: Number of hidden layers for MLP (default is 5).
    :return: Trained MLP classifier with best parameters.
    """
    print("Training MLP...")
    mlp_clf = MLPClassifier(random_state=17)
    # Set grid search for different layer configurations
    if num_layers == 1:
        param_grid = {"hidden_layer_sizes": [(i,) for i in np.arange(10, 100, 20)]}
    elif num_layers == 2:
        param_grid = {"hidden_layer_sizes": [(i, i) for i in np.arange(10, 100, 20)]}
    elif num_layers == 3:
        param_grid = {"hidden_layer_sizes": [(i, i, i) for i in np.arange(10, 100, 20)]}
    elif num_layers == 5:
        param_grid = {
            "hidden_layer_sizes": [(i, i, i, i, i) for i in np.arange(10, 100, 20)]
        }

    mlp_gscv = GridSearchCV(mlp_clf, param_grid, cv=cv, scoring="balanced_accuracy")
    mlp_gscv.fit(X, y)
    print(
        f"MLP with {num_layers} hidden layer: best balanced accuracy of {mlp_gscv.best_score_:.4f}, for a number of neurons per layer: {mlp_gscv.best_params_}."
    )
    return mlp_gscv


def get_3D_plot_data(plot_type: str):
    """
    Generate a 3D scatter plot for ChemBERTa ligand or ESM2 protein embeddings.

    :param plot_type: Type of plot ('ligands' or 'proteins').
    :return: Plotly figure with 3D scatter plot.
    :raises: NotImplementedError if plot_type is not 'ligands' or 'proteins'.
    """
    # Set plot parameters based on the plot type
    if plot_type == "ligands":
        file_path = "data/merged_umap_df.csv"
        labels = "UMAP"
        title = "3D KMeans Clustering of ChemBERTa ligand embeddings"
        dotsize = 6
    elif plot_type == "proteins":
        file_path = "data/protein_ligand_matched.csv"
        labels = "Protein UMAP"
        title = "Clustering of ESM2 protein embeddings, colored by ligand group"
        dotsize = 3
    else:
        raise NotImplementedError(f"{plot_type} is not implemented. Please use 'ligands' or 'proteins'.")

    # Load data and create the 3D scatter plot
    data = pd.read_csv(file_path)
    fig = px.scatter_3d(
        data,
        x=f"{labels} 1",
        y=f"{labels} 2",
        z=f"{labels} 3",
        color="class",
        title=title,
        hover_data={f"{labels} 1": False, f"{labels} 2": False, f"{labels} 3": False, "class": True},
    )
    fig.update_traces(marker=dict(size=dotsize))

    # Customize layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="UMAP 1", showgrid=True, zeroline=True),
            yaxis=dict(title="UMAP 2", showgrid=True, zeroline=True),
            zaxis=dict(title="UMAP 3", showgrid=True, zeroline=True),
            aspectmode="cube"
        ),
        legend_title_text="Class",
    )

    return fig