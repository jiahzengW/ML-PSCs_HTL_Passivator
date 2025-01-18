import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Function to initialize multiple machine learning models for regression tasks.
def initialize_models():
    """
    Initializes and returns a dictionary of regression models from various machine learning algorithms.

    Parameters:
    None

    Returns:
    dict: A dictionary where the keys are model names and the values are initialized model instances.
    """
    model_dict = {
        "RF": RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42),
        "XGB": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=30, max_depth=7, random_state=42),
        "LGBM": lgb.LGBMRegressor(n_estimators=130, max_depth=7, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=30, max_depth=7, random_state=42),
        "LR": LinearRegression(),
        "BR": BayesianRidge(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Support Vector Regression
    }
    print("Models initialized successfully.")
    return model_dict

# Function to evaluate models and compute R² scores on both training and test datasets.
def evaluate_models(models, X_tr, X_te, y_tr, y_te):
    """
    Evaluates the performance of multiple models by computing R² scores on both the training and test datasets.

    Parameters:
    models (dict): A dictionary of initialized models.
    X_tr (np.ndarray): The training feature matrix.
    X_te (np.ndarray): The test feature matrix.
    y_tr (np.ndarray): The training target values.
    y_te (np.ndarray): The test target values.

    Returns:
    tuple: Two dictionaries containing the R² scores for the training and test datasets.
    """
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
        except Exception as e:
            print(f"Error during training with {name}: {e}")
            train_scores[name] = np.nan
            test_scores[name] = np.nan
    return train_scores, test_scores

# Function to visualize the comparison of R² scores for training and test datasets.
def plot_r2_comparison(train_r2, test_r2):
    """
    Visualizes the R² scores for both training and test datasets using a bar plot.

    Parameters:
    train_r2 (dict): Dictionary containing R² scores for the training dataset.
    test_r2 (dict): Dictionary containing R² scores for the test dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(train_r2.keys())
    train_values = list(train_r2.values())
    test_values = list(test_r2.values())
    
    bar_width = 0.35
    index = np.arange(len(model_names))

    bar1 = ax.bar(index, train_values, bar_width, label="Train R²")
    bar2 = ax.bar(index + bar_width, test_values, bar_width, label="Test R²")

    ax.set_xlabel('Models')
    ax.set_ylabel('R² Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Function to perform hyperparameter tuning using grid search (optional, for models that support it).
def tune_hyperparameters(models, X_tr, y_tr):
    """
    Perform hyperparameter tuning using grid search for models that support it.

    Parameters:
    models (dict): A dictionary of initialized models.
    X_tr (np.ndarray): The training feature matrix.
    y_tr (np.ndarray): The training target values.

    Returns:
    dict: A dictionary with tuned models.
    """
    from sklearn.model_selection import GridSearchCV

    tuned_models = {}
    param_grid = {
        "RF": {'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20]},
        "XGB": {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [50, 100, 150]},
        "LGBM": {'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50, 100]},
        "GB": {'n_estimators': [100, 150], 'max_depth': [5, 10]},
        "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    }

    for model_name in models:
        if model_name in param_grid:
            print(f"Tuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(models[model_name], param_grid[model_name], cv=5, n_jobs=-1)
            grid_search.fit(X_tr, y_tr)
            tuned_models[model_name] = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            tuned_models[model_name] = models[model_name]  # No tuning for some models
            print(f"No tuning required for {model_name}.")
    return tuned_models

# Main function to orchestrate model training and evaluation.
def main():
    """
    Main function to execute the entire pipeline: data preparation, feature extraction, model training, evaluation, and plotting.
    """
    # Prepare data and features
    print("Loading and preparing the dataset...")
    data = load_data("molecule_data.csv")  # Load data
    molecules, features = prepare_data(data)  # Feature extraction

    # Initialize models
    models = initialize_models()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, data["Target"], test_size=0.3, random_state=42)

    # Perform hyperparameter tuning (optional)
    tuned_models = tune_hyperparameters(models, X_train, y_train)

    # Evaluate models
    print("Evaluating models...")
    train_r2, test_r2 = evaluate_models(tuned_models, X_train, X_test, y_train, y_test)

    # Visualize R² comparison
    print("Plotting R² comparison...")
    plot_r2_comparison(train_r2, test_r2)

if __name__ == "__main__":
    main()
