import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import numpy as np
graph_embedding_length = 600

def featurize_ChemBERTa(smiles_list, padding=True):
    embeddings_mean = torch.zeros(len(smiles_list), graph_embedding_length)
    print(embeddings_mean.shape)
    with torch.no_grad():
        for i, s in enumerate(smiles_list.values.tolist()):
            encoded_input = tokenizer(s[1], return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)

            embedding = torch.mean(model_output[0],1)
            embeddings_mean[i] = embedding
            
    return embeddings_mean.numpy()

df_smiles = pd.read_csv('data/all_unique_smiles.csv')
chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
# Code from https://www.kaggle.com/code/alexandervc/chembert2a-smiles-embeddings-for-beginners/notebook
chemberta.eval()
graph_embeddings = featurize_ChemBERTa(df_smiles['Ligand SMILES'])
print(graph_embeddings)
df = pd.DataFrame(graph_embeddings)
df.to_csv("all_graph_embeddings.csv", index=False)