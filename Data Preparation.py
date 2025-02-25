import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Function to load dataset from a specified CSV file.
def load_data(filepath):
    """
    Loads the data from the specified CSV file, handling potential errors if the file is not found.
    Parameters:
    filepath (str): The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

# Function to convert SMILES strings into RDKit molecular objects.
def convert_smiles_to_mol(smiles_list):
    """
    Converts a list of SMILES strings into RDKit molecular objects, filtering out invalid ones.

    Parameters:
    smiles_list (list): List of SMILES strings representing chemical compounds.

    Returns:
    list: A list of valid RDKit molecular objects.
    """
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_mols = [mol for mol in molecules if mol is not None]
    print(f"Successfully converted {len(valid_mols)} valid molecules from SMILES strings.")
    return valid_mols

# Function to handle missing values by applying mean imputation.
def impute_missing_values(data):
    """
    Imputes missing values in the dataset using the mean strategy for numerical columns.

    Parameters:
    data (pd.DataFrame): The input dataset with potential missing values.

    Returns:
    np.ndarray: The dataset with missing values imputed.
    """
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(data)
    print("Missing values imputed using the mean strategy.")
    return imputed_data

# Main function to prepare the dataset for model training.
def prepare_data(filepath):
    """
    Prepares the data by loading it from a file, converting SMILES strings, and imputing missing values.

    Parameters:
    filepath (str): Path to the input CSV file containing the dataset.

    Returns:
    tuple: A tuple containing the list of molecules and the imputed feature matrix.
    """
    data = load_data(filepath)
    if data is None:
        return None, None
    
    molecules = convert_smiles_to_mol(data["SMILES"])
    features = impute_missing_values(data.drop(columns=["SMILES"]))
    
    return molecules, features

# Test the preparation function with a sample dataset
if __name__ == "__main__":
    file_path = "molecule_data.csv"
    molecules, features = prepare_data(file_path)
    print(f"Data preparation completed with {len(molecules)} valid molecules.")
