# data_loading.py
import pandas as pd


def load_data(idea="idea2"):

    if idea == "idea2":
        filtered_best_ligand_ic50 = pd.read_csv("data/filtered_best_ligand_ic50.csv")
        unique_smiles_df = pd.DataFrame(
            {"Ligand SMILES": filtered_best_ligand_ic50["Ligand SMILES_x"].unique()}
        )
        embeddings = pd.read_csv("data/all_graph_embeddings.csv")
        return filtered_best_ligand_ic50, unique_smiles_df, embeddings
    # elif ...
