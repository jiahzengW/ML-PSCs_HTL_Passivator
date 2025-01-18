import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem import rdMolDescriptors

# Function to generate different molecular fingerprints (MACCS, ECFP, and Morgan).
def generate_fingerprints(molecules):
    """
    Generates molecular fingerprints for each molecule using MACCS, ECFP, and Morgan fingerprinting methods.

    Parameters:
    molecules (list): List of RDKit molecular objects.

    Returns:
    tuple: Tuple containing three numpy arrays: MACCS fingerprints, ECFP fingerprints, and Morgan fingerprints.
    """
    maccs_fp, ecfp_fp, morgan_fp = [], [], []

    # Generate MACCS fingerprints
    for idx, molecule in enumerate(molecules):
        try:
            maccs = list(MACCSkeys.GenMACCSKeys(molecule))
            maccs_fp.append(maccs)
        except Exception:
            print(f"MACCS fingerprint generation failed for molecule {idx}")
    
    # Generate ECFP (Extended Connectivity Fingerprint) fingerprints
    for idx, molecule in enumerate(molecules):
        try:
            ecfp = list(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 1024))
            ecfp_fp.append(ecfp)
        except Exception:
            print(f"ECFP fingerprint generation failed for molecule {idx}")

    # Generate Morgan fingerprints
    for idx, molecule in enumerate(molecules):
        try:
            morgan = list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024))
            morgan_fp.append(morgan)
        except Exception:
            print(f"Morgan fingerprint generation failed for molecule {idx}")

    return np.array(maccs_fp), np.array(ecfp_fp), np.array(morgan_fp)

# Function to extract molecular features based on RDKit descriptors.
def extract_features(molecules):
    """
    Extracts a set of molecular features from the given molecules using RDKit descriptors.

    Parameters:
    molecules (list): List of RDKit molecular objects.

    Returns:
    np.ndarray: A matrix of extracted molecular features.
    """
    features_list = []
    for idx, mol in enumerate(molecules):
        feature_set = []
        try:
            # Extract various descriptors for each molecule
            feature_set.append(Descriptors.ExactMolWt(mol))  # Molecular weight
            feature_set.append(rdMolDescriptors.CalcTPSA(mol))  # Topological Polar Surface Area (TPSA)
            feature_set.append(Descriptors.NumHDonors(mol))  # Number of hydrogen bond donors
            feature_set.append(Descriptors.NumHAcceptors(mol))  # Number of hydrogen bond acceptors
            
            # Estimated melting point based on molecular weight and TPSA
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            melt_pt = 0.5 * tpsa + 0.2 * mw
            feature_set.append(melt_pt)
            
            # Steric hindrance based on molecular volume and rotatable bonds
            mol_volume = Descriptors.MolMR(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            steric = mol_volume + rot_bonds
            feature_set.append(steric)
            
            feature_set.append(Descriptors.MolLogP(mol))  # LogP value (lipophilicity)
            feature_set.append(Descriptors.NumValenceElectrons(mol))  # Number of valence electrons
            
            features_list.append(feature_set)
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
    
    return np.array(features_list)

# Function to perform feature extraction on a list of molecules.
def feature_engineering(molecules):
    """
    Extracts relevant features from the given molecules, combining both molecular descriptors and fingerprints.

    Parameters:
    molecules (list): List of RDKit molecular objects.

    Returns:
    np.ndarray: A matrix of features derived from the molecules.
    """
    print("Extracting features from molecules...")
    features = extract_features(molecules)
    return features

# Test the feature extraction function with sample molecules
if __name__ == "__main__":
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ["CCO", "CCN", "CNC"]]
    features = feature_engineering(molecules)
    print(f"Feature extraction completed with {features.shape[0]} molecules and {features.shape[1]} features.")
