import os

import numpy as np
import pandas as pd
import torch
import igraph as ig

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm

def extract_data(df):
    """
    Extract proteins, ligands, and IC50 values from the DataFrame.
    
    :param df: DataFrame containing the columns 'BindingDB Target Chain Sequence', 'Ligand SMILES', and 'IC50 (nM)'.
    :return: Tuple of proteins, ligands, and IC50 values.
    """
    proteins = list(df['BindingDB Target Chain Sequence'])
    ligands = list(df['Ligand SMILES'])
    ic50 = list(df['IC50 (nM)'])
    return proteins, ligands, ic50


def calculate_interaction_strengths(proteins, ligands, ic50):
    """
    Calculate interaction strengths using IC50 values.
    
    :param proteins: List of protein sequences.
    :param ligands: List of ligand SMILES strings.
    :param ic50: List of IC50 values.
    :return: List of interactions and edge weights.
    """
    interactions = [(protein, ligand, 1 / np.log(ic + 1.05)) for protein, ligand, ic in zip(proteins, ligands, ic50)]
    edge_weights = [float(interaction[2]) for interaction in interactions]
    return interactions, edge_weights


def create_bipartite_graph(proteins, ligands, interactions, edge_weights):
    """
    Create a bipartite graph using protein-ligand interaction data.
    
    :param proteins: List of protein sequences.
    :param ligands: List of ligand SMILES strings.
    :param interactions: List of protein-ligand interactions.
    :param edge_weights: List of interaction strengths.
    :return: The created igraph bipartite graph.
    """
    unique_vertices = list(set(proteins + ligands))
    vertex_indices = {name: idx for idx, name in enumerate(unique_vertices)}
    B = ig.Graph(n=len(unique_vertices))

    B.vs["name"] = unique_vertices
    edge_list = [(vertex_indices[protein], vertex_indices[ligand]) for protein, ligand, _ in interactions]
    B.add_edges(edge_list)
    B.es["weight"] = edge_weights
    
    return B, vertex_indices


def set_vertex_types(B, proteins):
    """
    Set the type of vertices in the bipartite graph (proteins vs ligands).
    
    :param B: The igraph bipartite graph.
    :param proteins: List of proteins.
    """
    proteins_set = set(proteins)
    B.vs["type"] = [v in proteins_set for v in B.vs["name"]]


def build_bipartite_graph(df):
    """
    Build a bipartite graph from the given DataFrame containing protein-ligand interactions.
    
    :param df: DataFrame containing the protein-ligand interaction data.
    :return B: The bipartite graph.
    """
    proteins, ligands, ic50 = extract_data(df)
    interactions, edge_weights = calculate_interaction_strengths(proteins, ligands, ic50)
    B, vertex_indices = create_bipartite_graph(proteins, ligands, interactions, edge_weights)
    set_vertex_types(B, proteins)
    
    return B


def build_sub_bipartite_graph(parent_graph, df):
    """
    Build a sub-graph with specific protein-ligand interactions from a parent graph
    
    :param parent_graph: Bipartite graph from which to extract the specific protein-ligand pairs
    :param df: DataFrame containing the specific protein-ligand pairs
    :return sub_B: Sub-part of the parent bipartite graph for the specific protein-ligand pairs
    """
    subset_proteins = list(df['BindingDB Target Chain Sequence'])
    subset_ligands = list(df['Ligand SMILES'])
    subset_vertex_indices = parent_graph.vs.select(name_in=subset_proteins + subset_ligands).indices
    sub_B = parent_graph.subgraph(subset_vertex_indices)
    
    return sub_B


def plot_bipartite_graph(bipartite_graph, save_path=None):
    """
    Plot bipartite graph 
    
    :param bipartite_graph: bipartite graph to plot, nodes are proteins and ligands
    :param save_path: if specified, the path to save the generated plot.
    """
    layout = bipartite_graph.layout_bipartite(types=bipartite_graph.vs["type"])
    layout_coords = [(x, -y) for x, y in layout.coords]

    plot = ig.plot(
        bipartite_graph,
        layout=layout_coords,
        vertex_size=8,
        vertex_color=["lightblue" if t else "lightgreen" for t in bipartite_graph.vs["type"]],
        edge_width=[w / 20 for w in bipartite_graph.es["weight"]],
        bbox=(1100, 600),
        margin=70,
        vertex_label=None
    )
    plot.save("temp_bipartite_graph.png")

    img = Image.open("temp_bipartite_graph.png") # required as iGraph doesn't supports adding titles to plots

    # Use Matplotlib to add a title and legend
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Bipartite graph: proteins and ligands", fontsize=16)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Proteins', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Ligands', markerfacecolor='lightgreen', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=False)

    if save_path:
        plt.savefig(os.path.join(save_path, "bipartite_graph.png"), dpi=600, bbox_inches="tight")
    
    plt.show()

    os.remove("temp_bipartite_graph.png")


def build_unipartite_graph(bipartite_graph, entity_type):
    """
    Project the bipartite graph onto a unipartite graph 
    
    :param bipartite_graph: bipartite graph to plot, nodes are proteins and ligands
    :param entity_type: the entity type to choose to project the bipartite graph
    :return unipartite_graph: The unipartite graph.
    """
    unipartite_graph = ig.Graph()
    if entity_type == 'ligand':
        indices = [i for i, v in enumerate(bipartite_graph.vs["type"]) if not v] # non-protein vertices
    else:
        indices = [i for i, v in enumerate(bipartite_graph.vs["type"]) if not v] # protein vertices

    unipartite_graph.add_vertices(len(indices))
    entity_to_unipartite = {indices[i]: i for i in range(len(indices))}

    # Seek for neighbor-sharing nodes
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            # Get the two nodes
            node_i = indices[i]
            node_j = indices[j]
            
            # Find common neighbors between node_i and node_j
            neighbors_i = set(bipartite_graph.neighbors(node_i))
            neighbors_j = set(bipartite_graph.neighbors(node_j))
            
            common_entity = neighbors_i.intersection(neighbors_j)
            
            # Compute the connection strength as the sum of weights w.r.t. the shared entity
            if common_entity:
                total_weight = 0
                for entity in common_entity:
                    # Extract the weight connecting the two neighbors in the bipartite graph
                    edge_id_i = bipartite_graph.get_eid(node_i, entity, directed=False, error=False)
                    edge_id_j = bipartite_graph.get_eid(node_j, entity, directed=False, error=False)
                    weight_i = bipartite_graph.es[edge_id_i]["weight"]
                    weight_j = bipartite_graph.es[edge_id_j]["weight"]
                    # Sum the weights
                    total_weight += weight_i + weight_j
                
                # Add an edge between the nodes in the unipartite graph with the calculated weight
                unipartite_graph.add_edge(entity_to_unipartite[node_i], entity_to_unipartite[node_j], weight=total_weight)
    
    return unipartite_graph


def plot_unipartite_graph(unipartite_graph, entity_type, save_path=None):
    """
    Plot unipartite graph 
    
    :param unipartite_graph: unipartite graph to plot, nodes are proteins and ligands
    :param entity_type: the entity type to whose unipartite graph is to be plotted
    :param save_path: if specified, the path to save the generated plot.
    """
    # Specify graph layout
    unipartite_layout = unipartite_graph.layout("drl")
    color = 'lightblue' if entity_type=='protein' else 'lightgreen'

    plot = ig.plot(
        unipartite_graph,
        layout=unipartite_layout,
        vertex_size=8,
        vertex_color=color,
        edge_width=[w["weight"] / 10 for w in unipartite_graph.es],
        bbox=(1000, 600),
        margin=10,
        vertex_label=None
    )
    plot.save("temp_unipartite_graph.png")

    img = Image.open("temp_unipartite_graph.png") # required as iGraph doesn't supports adding titles to plots

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Unipartite graph: {entity_type}s", fontsize=16)

    if save_path:
        plt.savefig(os.path.join(save_path, f"unipartite_graph_{entity_type}.png"), dpi=600, bbox_inches="tight")
    
    plt.show()

    os.remove("temp_unipartite_graph.png")


def load_graph_weights(bipartite_graph, unipartite_graph, entity_type):
    """
    Extract edge weights from a unipartite graph and map them to sequences from a bipartite graph.

    :param bipartite_graph: Bipartite graph containing ligand and protein information.
    :param unipartite_graph: Unipartite graph with edges and weights to process.
    :param entity_type: Type of entity ('ligand' or 'protein') to extract weights for.
    :return: DataFrame with pairs of entities and their corresponding edge weights.
    """
    df_weights = []

    # Determine indices of vertices based on entity type
    if entity_type == 'ligand':
        indices = [i for i, v in enumerate(bipartite_graph.vs["type"]) if not v]  # Non-protein vertices
    else:
        indices = [i for i, v in enumerate(bipartite_graph.vs["type"]) if v]  # Protein vertices

    # Iterate through edges in the unipartite graph
    for edge in unipartite_graph.es:
        node_i_index = edge.source
        node_j_index = edge.target

        # Map unipartite graph nodes to bipartite graph sequences
        node_i_sequence = bipartite_graph.vs[indices[node_i_index]]["name"]
        node_j_sequence = bipartite_graph.vs[indices[node_j_index]]["name"]
        
        edge_weight = edge["weight"]
        
        # Append data for the current edge
        df_weights.append({
            f"{entity_type} 1": node_i_sequence,
            f"{entity_type} 2": node_j_sequence,
            "Weight": edge_weight
        })

    df_weights = pd.DataFrame(df_weights)

    return df_weights


def load_chemberta(df, filepath):
    """
    Load ChemBERTa embeddings as a formatted DataFrame

    :param df: DataFrame containing a 'Ligand SMILES' column.
    :param filepath: Path to the file containing ChemBERTa embeddings.
    :return: DataFrame with ligand SMILES and their ChemBERTa embeddings.
    """
    chemberta_df = pd.read_csv(filepath)

    chemberta_df['Ligand SMILES'] = df['Ligand SMILES'].unique()
    chemberta_df.set_index('Ligand SMILES', inplace=True, drop=True)

    ligand_emb = []
    for ligand_smiles, row in chemberta_df.iterrows():
        ligand_emb.append([ligand_smiles, np.array(row)])  # Convert row to NumPy array

    columns = ['Ligand SMILES', 'ChemBERTa Embedding']
    chemberta_df = pd.DataFrame(ligand_emb, columns=columns)

    return chemberta_df


def compute_cs_protein(df_embeddings, df_weights):
    """
    Compute pairwise cosine similarities between protein embeddings from ESM2.

    :param df_embeddings: DataFrame with protein sequences and their ESM2 embeddings.
    :param df_weights: DataFrame with protein pairs for which to compute similarities.
    :return: Array of cosine similarity scores for each protein pair.
    """
    # Reformat tensor embeddings into numpy array
    df_embeddings['ESM2 Embedding'] = df_embeddings['ESM2 Embedding'].apply(lambda x: eval("torch." + x))

    cosine_sim_protein = np.empty((len(df_weights), ))

    # Iterate over protein pairs in weights DataFrame
    for idx, row in df_weights.iterrows():
        # Retrieve embeddings for each protein
        esm2_1 = df_embeddings.loc[df_embeddings['BindingDB Target Chain Sequence'] == row['protein 1']]['ESM2 Embedding'].values
        esm2_2 = df_embeddings.loc[df_embeddings['BindingDB Target Chain Sequence'] == row['protein 2']]['ESM2 Embedding'].values
        
        # Compute cosine similarity
        cosine_sim_protein[idx] = cosine_similarity(esm2_1[0].numpy(), esm2_2[0].numpy())[0, 0]
    
    return cosine_sim_protein


def compute_cs_ligand(df_embeddings, df_weights):
    """
    Compute pairwise cosine similarities between ligand embeddings from ChemBERTa.

    :param df_embeddings: DataFrame with ligand SMILES and their ChemBERTa embeddings.
    :param df_weights: DataFrame with ligand pairs for which to compute similarities.
    :return: Array of cosine similarity scores for each ligand pair.
    """
    cosine_sim = np.empty((len(df_weights), ))

    # Iterate over ligand pairs in weights DataFrame
    for idx, row in df_weights.iterrows():
        # Retrieve embeddings for each ligand
        chemberta_1 = df_embeddings.loc[df_embeddings['Ligand SMILES'] == row['ligand 1']]['ChemBERTa Embedding'].values[0]
        chemberta_2 = df_embeddings.loc[df_embeddings['Ligand SMILES'] == row['ligand 2']]['ChemBERTa Embedding'].values[0]

        # Compute cosine similarity
        cosine_sim[idx] = cosine_similarity(chemberta_1.reshape(1, -1), chemberta_2.reshape(1, -1))[0, 0]
    
    return cosine_sim


def plot_regression_model(df_weights, entity_type, save_path=None):
    """
    Plot a regression model for the relationship between graph edge weights and cosine similarity of embeddings.

    :param df_weights: DataFrame containing graph edge weights and cosine similarity scores.
    :param entity_type: Type of entity ('ligand' or 'protein') to determine embedding type.
    :param savepath: Optional path to save the generated plot.
    """
    # Determine embedding type based on entity tyoe
    if entity_type == 'ligand':
        embed_name = 'ChemBERTa'
    else:
        embed_name = 'ESM2'
    
    # Fit linear regression model
    X = sm.add_constant(df_weights['Weight'])
    model = sm.OLS(df_weights[f'{embed_name} cosine similarity'], X).fit()
    
    # Model evaluation
    r_squared = model.rsquared

    # Create regression plot with adapted colors
    col = 'lightblue' if entity_type == 'protein' else 'lightgreen'
    col_line = 'blue' if entity_type == 'protein' else 'green'
    sns.regplot(
        data=df_weights, 
        x='Weight', 
        y=f'{embed_name} cosine similarity', 
        ci=95, 
        scatter_kws={'s': 1, 'color': col}, 
        line_kws={'color': col_line, 'linewidth': 1}
    )

    # Add legend for confidence interval and R²
    ci_patch = mpatches.Patch(color=col, alpha=0.3, label='95% CI')
    r2_line = mlines.Line2D([], [], color=col_line, label=f'R²: {r_squared:.3f}', linewidth=2)
    plt.legend(handles=[ci_patch, r2_line], loc='lower right', frameon=False);
    
    print(model.summary())

    if save_path:
        plt.savefig(os.path.join(save_path, f'regression_plot_{entity_type}.png'), dpi=600, bbox_inches='tight')
