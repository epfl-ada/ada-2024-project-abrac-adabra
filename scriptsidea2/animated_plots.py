import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def get_3D_plot_data(plot_type: str):
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
        raise NotImplemented(
            f"{plot_type} is not implemented. Please use 'ligands' or 'proteins'."
        )

    data = pd.read_csv(file_path)

    fig = px.scatter_3d(
        data,
        x=f"{labels} 1",
        y=f"{labels} 2",
        z=f"{labels} 3",
        color="class",
        title=title,
        hover_data={
            f"{labels} 1": False,
            f"{labels} 2": False,
            f"{labels} 3": False,
            "class": True,
        },
    )
    fig.update_traces(marker=dict(size=dotsize))
    fig.update_layout(
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        legend_title_text="Class",
    )

    return fig


def main(plot_name: str):
    if plot_name == "combined_embeddings":
        fig_ligands = get_3D_plot_data("ligands")
        fig_proteins = get_3D_plot_data("proteins")

        # combine the figures to make subplots
        combined_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "3D KMeans Clustering of ChemBERTa ligand embeddings",
                "Clustering of ESM2 protein embeddings, colored by ligand group",
            ),
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
        )

        for trace in fig_ligands["data"]:
            combined_fig.add_trace(trace, row=1, col=1)

        for trace in fig_proteins["data"]:
            combined_fig.add_trace(trace, row=1, col=2)

        combined_fig.update_layout(
            title_text="",
            height=500,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=50),
            title_x=0.5,
            showlegend=False,
        )

        combined_fig.layout.annotations[0].update(text="ChemBERTa ligand embeddings")
        combined_fig.layout.annotations[1].update(
            text="ESM2 protein embeddings (colored by ligand group)"
        )
        combined_fig.layout.scene1.xaxis.title.text = "UMAP 1"
        combined_fig.layout.scene1.yaxis.title.text = "UMAP 2"
        combined_fig.layout.scene1.zaxis.title.text = "UMAP 3"
        combined_fig.layout.scene2.xaxis.title.text = "UMAP 1"
        combined_fig.layout.scene2.yaxis.title.text = "UMAP 2"
        combined_fig.layout.scene2.zaxis.title.text = "UMAP 3"

        pio.write_html(
            combined_fig,
            file="data/plots/3d_combined_clustering_plot.html",
            auto_open=True,
        )
    else:
        raise NotImplementedError(f"{plot_name} is not implemented.")


main("combined_embeddings")
