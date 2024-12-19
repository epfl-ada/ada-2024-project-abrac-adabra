import warnings
import requests
import pandas as pd 

data_path = '../../data/BindingDB/BindingDB_All.tsv'

def filter_out_comparator(s):
    """
    Filtering out values that do not match an exact IC50 or Ki value
    :param s: IC50 or Ki value
    :return bool: False if the IC50 or Ki entry contains a mathematical symbol, True otherwise
    """
    if type(s)==str and ('>' in s or '<' in s):
        return False
    else:
        return True

def get_protein_name(entry_name):
    """
    Retrieveing protein name from UniProt
    :param entry_name: UniProt name of the Target Chain in BindingDB
    :return protein_name: Protein name of the mutant
    """
    url = f"https://rest.uniprot.org/uniprotkb/{entry_name}.json"
    
    response = requests.get(url)
    
    # Check if the response was successful
    if response.status_code == 200:
        data = response.json()
        # Extract the protein name
        protein_name = data['proteinDescription']['recommendedName']['fullName']['value']
        return protein_name
    else:
        print(f"Entry '{entry_name}' not found or an error occurred.")
        return None
    
print('Loading the Data')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Specify dtype option on import or set low_memory=False.*")
    df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip');

print('Filtering the Data')
# Focusing on proteins with a single chain
df = df[df['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1]

# Selecting a subset of columns of interest
useful_cols = ['BindingDB Reactant_set_id', 'Ligand SMILES', 'Target Name', 'IC50 (nM)', 
'BindingDB Target Chain Sequence', 'UniProt (SwissProt) Entry Name of Target Chain']

filtered_df = df[useful_cols]
filtered_df.set_index('BindingDB Reactant_set_id', inplace=True)

# Creating a filtered dataframe for IC50 (nM) (keeping only exact values)
clean_df = filtered_df.copy()
clean_df = clean_df[clean_df['IC50 (nM)'].apply(filter_out_comparator)]
clean_df['IC50 (nM)'] = pd.to_numeric(clean_df['IC50 (nM)'], errors='coerce')
clean_df = clean_df.dropna(subset=['IC50 (nM)'])

# Merging all the different entries using the median of each group
merged_df = clean_df.groupby(['Ligand SMILES', 'Target Name', 'BindingDB Target Chain Sequence', 
                              'UniProt (SwissProt) Entry Name of Target Chain'])['IC50 (nM)'].median().reset_index()

print('Saving the processed data')
# Exporting the dataframe to a .csv
merged_df.to_csv('../data/merged.csv')

print('Processing the data further for mutant analysis')
# Filtering out proteins that do not have any mutant
mutants_filtered_df = merged_df.groupby(['Ligand SMILES', 'UniProt (SwissProt) Entry Name of Target Chain']).filter(lambda x: len(x) >= 2)

# Creating a new dataframe with the filtered entries
merged_mutants_filtered_df = mutants_filtered_df.groupby(['Ligand SMILES', 'UniProt (SwissProt) Entry Name of Target Chain']).apply(lambda x: pd.Series({
    'Target Names': list(x['Target Name']),
    'BindingDB Target Chain Sequence': list(x['BindingDB Target Chain Sequence'])})).reset_index()

# Applying the "get_protein_name" function to the dataframe to retrieve the wild type Name
protein_names = {entry: get_protein_name(entry) for entry in merged_mutants_filtered_df['UniProt (SwissProt) Entry Name of Target Chain'].unique()}
merged_mutants_filtered_df['WT Target Name'] = merged_mutants_filtered_df.apply(lambda x: protein_names.get(x['UniProt (SwissProt) Entry Name of Target Chain'], None) 
                                 if protein_names.get(x['UniProt (SwissProt) Entry Name of Target Chain'], None) in x['Target Names'] else None, axis=1)
merged_mutants_filtered_df.dropna(subset=['WT Target Name'], inplace=True)


# Exporting the dataframe to a csv file
print('Saving the mutants data')
merged_mutants_filtered_df.to_csv('../data/mutants.csv')