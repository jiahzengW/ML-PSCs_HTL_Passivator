import shap
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

def initialize_shap(model, test_data):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(test_data)
    return explainer, shap_vals

def prepare_features(test_df, feature_mapping, selected_features):
    renamed = test_df.rename(columns=feature_mapping)
    ordered = renamed[selected_features]
    return ordered

def extract_shap_values(shap_vals, original_features, selected_features):
    indices = [original_features.index(name) for name in selected_features if name in original_features]
    selected_shap = shap_vals[:, indices]
    shap_df = pd.DataFrame(selected_shap, columns=selected_features)
    return shap_df

def plot_summary(shap_vals, features_df, save_path):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.figure(figsize=(16, 8))
    shap.summary_plot(shap_vals, features_df, plot_type="dot", show=False)
    plt.title('SHAP Summary Plot', fontsize=28, fontweight='bold', pad=15)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.tick_params(axis='both', labelsize=26, which='major', labelrotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)
    plt.show()

def plot_dependence(shap_vals, features_df, keys, save_dir):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    for key in keys:
        plt.figure(figsize=(6, 6))
        shap.dependence_plot(key, shap_vals, features_df, display_features=features_df, show=False)
        plt.title(f'SHAP Dependence Plot for {key}', fontsize=20, fontweight='bold', pad=15)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color('black')
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.show()

def plot_pair_dependence(shap_vals, features_df, pairs, save_dir):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    for x_feat, y_feat in pairs:
        plt.figure(figsize=(6, 6))
        shap.dependence_plot(x_feat, shap_vals, features_df, display_features=features_df, interaction_index=y_feat, show=False)
        plt.title(f'SHAP Dependence Plot for {x_feat} vs {y_feat}', fontsize=24, fontweight='bold', pad=15)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color('black')
        plt.tick_params(axis='both', labelsize=24)
        plt.tight_layout()
        plt.show()

def plot_force(explainer, shap_vals, features_df, sample_count=3):
    for i in range(sample_count):
        plt.figure(figsize=(10, 2))
        shap.force_plot(explainer.expected_value, shap_vals[i, :], features_df.iloc[i, :], matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot for Sample {i+1}', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()

def main(model, test_df, feature_mapping, selected_features, key_features, feature_pairs, save_dir):
    ordered_features = prepare_features(test_df, feature_mapping, selected_features)
    explainer, shap_vals = initialize_shap(model, ordered_features)
    shap_df = extract_shap_values(shap_vals, test_df.columns.tolist(), selected_features)


    plot_summary(shap_vals, ordered_features, summary_path)
    
    plot_dependence(shap_vals, ordered_features, key_features, save_dir)
    
    plot_pair_dependence(shap_vals, ordered_features, feature_pairs, save_dir)
    
    plot_force(explainer, shap_vals, ordered_features, sample_count=3, save_dir=save_dir)

if __name__ == "__main__":
    main(
        model=xgb_model,
        test_df=test_data,
        feature_mapping=feature_mapping,
        selected_features=selected_features,
        key_features=key_features,
        feature_pairs=feature_pairs,
        save_dir=save_directory
    )
