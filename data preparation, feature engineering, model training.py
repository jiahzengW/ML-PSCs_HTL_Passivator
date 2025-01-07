import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def generate_fingerprints(molecules):
    maccs_fp, ecfp_fp, morgan_fp = [], [], []
    valid_maccs, valid_ecfp, valid_morgan = [], [], []

    for idx, molecule in enumerate(molecules):
        try:
            maccs = list(MACCSkeys.GenMACCSKeys(molecule))
            maccs_fp.append(maccs)
            valid_maccs.append(idx)
        except Exception:
            print(f"MACCS generation failed for molecule {idx}")

    for idx, molecule in enumerate(molecules):
        try:
            ecfp = list(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 1024))
            ecfp_fp.append(ecfp)
            valid_ecfp.append(idx)
        except Exception:
            print(f"ECFP generation failed for molecule {idx}")

    for idx, molecule in enumerate(molecules):
        try:
            morgan = list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024))
            morgan_fp.append(morgan)
            valid_morgan.append(idx)
        except Exception:
            print(f"Morgan fingerprint generation failed for molecule {idx}")

    return (
        np.array(maccs_fp),
        np.array(ecfp_fp),
        np.array(morgan_fp),
        valid_maccs,
        valid_ecfp,
        valid_morgan
    )

def fingerprints_to_excel(maccs, ecfp, morgan, paths):
    maccs_df = pd.DataFrame(maccs)
    ecfp_df = pd.DataFrame(ecfp)
    morgan_df = pd.DataFrame(morgan)
    
    maccs_df.to_excel(paths[0], index=False)
    ecfp_df.to_excel(paths[1], index=False)
    morgan_df.to_excel(paths[2], index=False)
    
def tally_doubles(molecule):
    count = 0
    for bond in molecule.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            count += 1
    return count

molecules = [Chem.MolFromSmiles(smile) for smile in data["SMILES"]]
features_list = []
processed_indices = []
total = len(molecules)

for idx in range(total):
    mol = molecules[idx]
    
    # Progress update
    print(f"Processing molecule {idx+1}/{total} ({(idx+1)/total*100:.2f}%)")
    
    # Add explicit hydrogens
    mol = Chem.AddHs(mol)

    feature_set = []
    
    try:
        # Exact molecular weight
        feature_set.append(Descriptors.ExactMolWt(mol))
        
        # Topological polar surface area
        feature_set.append(rdMolDescriptors.CalcTPSA(mol))
        
        # Number of H-bond donors
        feature_set.append(Descriptors.NumHDonors(mol))
        
        # Number of H-bond acceptors
        feature_set.append(Descriptors.NumHAcceptors(mol)
        
        # Estimated melting point
        mw = Descriptors.MolWt(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        melt_pt = 0.5 * tpsa + 0.2 * mw
        feature_set.append(melt_pt)
        
        # Steric hindrance
        mol_volume = Descriptors.MolMR(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        steric = mol_volume + rot_bonds
        feature_set.append(steric)
        
        # LogP value
        feature_set.append(Descriptors.MolLogP(mol))
        
        # Valence electrons
        feature_set.append(Descriptors.NumValenceElectrons(mol))
          
    features_list.append(feature_set)
    processed_indices.append(idx)

    
def split_dataset(features, targets, test_size=0.3, seed=42):
    return train_test_split(features, targets, test_size=test_size, random_state=seed)

def initialize_models():
    model_dict = {
        "RF": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        "XGB": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=5, random_state=42),
        "LGBM": lgb.LGBMRegressor(n_estimators=50, max_depth=5, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        "LR": LinearRegression(),
        "BR": BayesianRidge(),
        "KNN": KNeighborsRegressor(n_neighbors=3),
    }
    return model_dict

def evaluate_models(models, X_tr, X_te, y_tr, y_te):
    train_scores = {}
    test_scores = {}
    for name, mdl in models.items():
        try:
            mdl.fit(X_tr, y_tr)
            train_pred = mdl.predict(X_tr)
            test_pred = mdl.predict(X_te)
            train_r2 = r2_score(y_tr, train_pred)
            test_r2 = r2_score(y_te, test_pred)
            train_scores[name] = train_r2
            test_scores[name] = test_r2
            print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        except Exception:
            print(f"{name} encountered an error.")
            train_scores[name] = np.nan
            test_scores[name] = np.nan
    return train_scores, test_scores


def main():
    configure_plot()
    features, targets = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_dataset(features, targets)
    models = initialize_models()
    train_r2, test_r2 = evaluate_models(models, X_train, X_test, y_train, y_test)
    plot_r2_comparison(train_r2, test_r2)

if __name__ == "__main__":
    main()

