import argparse
import pandas as pd
import numpy as np
import sys
import os
import json
import contextlib
import warnings
from joblib import Parallel, delayed, dump
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress standard output and error
class SuppressOutput(contextlib.AbstractContextManager):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def regression(inp, prefix, selected_models):
    # Remove directory creation since we are saving files locally with prefix

    # Load data
    data = pd.read_csv(inp)

    # Data preprocessing
    data = data.dropna()

    # Check required columns
    required_columns = ['SampleID', 'Label']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Input data must contain '{col}' column.")

    # Extract SampleID
    sample_ids = data['SampleID']

    # Extract features and target variable, ensuring y is one-dimensional
    X = data.drop(columns=['SampleID', 'Label'])
    y = data['Label'].astype(float).values.ravel()

    # Ensure feature matrix and target variable have the same number of samples
    assert X.shape[0] == y.shape[0], "Number of samples in features and target variable do not match."

    # Set global random seed for reproducibility
    RANDOM_SEED = 1234
    np.random.seed(RANDOM_SEED)

    # Prepare regression models with 'reg' suffix in names
    regressors = {
        'Neural_Network_reg': MLPRegressor(max_iter=200000, random_state=RANDOM_SEED),
        'Random_Forest_reg': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        'SVM_reg': SVR(),
        'XGBoost_reg': XGBRegressor(eval_metric='rmse', random_state=RANDOM_SEED, verbosity=0, n_jobs=-1),
        'PLS_reg': PLSRegression(),
        'KNN_reg': KNeighborsRegressor(n_jobs=-1),
        'LightGBM_reg': LGBMRegressor(random_state=RANDOM_SEED, force_col_wise=True, verbosity=-1, n_jobs=-1)
    }

    # Filter selected models
    selected_regressors = {name: regressors[name] for name in selected_models if name in regressors}
    if not selected_regressors:
        raise ValueError("No valid models selected. Please check the model names.")

    # Define hyperparameter search spaces for each model
    def get_param_distributions(name, num_features):
        if name == 'Neural_Network_reg':
            return {
                'hidden_layer_sizes': optuna.distributions.IntDistribution(low=10, high=200),
                'alpha': optuna.distributions.FloatDistribution(low=1e-4, high=1e-1),
                'learning_rate_init': optuna.distributions.FloatDistribution(low=1e-4, high=1e-1)
            }
        elif name == 'Random_Forest_reg':
            return {
                'n_estimators': optuna.distributions.IntDistribution(low=100, high=1000),
                'max_depth': optuna.distributions.IntDistribution(low=5, high=50),
                'max_features': optuna.distributions.CategoricalDistribution(choices=['sqrt', 'log2', None]),
                'min_samples_split': optuna.distributions.IntDistribution(low=2, high=20),
                'min_samples_leaf': optuna.distributions.IntDistribution(low=1, high=20)
            }
        elif name == 'SVM_reg':
            return {
                'C': optuna.distributions.FloatDistribution(low=1e-3, high=1e3, log=True),
                'gamma': optuna.distributions.FloatDistribution(low=1e-4, high=1e1, log=True),
                'epsilon': optuna.distributions.FloatDistribution(low=1e-4, high=1e1, log=True)
            }
        elif name == 'XGBoost_reg':
            return {
                'n_estimators': optuna.distributions.IntDistribution(low=50, high=500),
                'max_depth': optuna.distributions.IntDistribution(low=3, high=15),
                'learning_rate': optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
                'subsample': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
                'colsample_bytree': optuna.distributions.FloatDistribution(low=0.5, high=1.0)
            }
        elif name == 'PLS_reg':
            upper = max(2, (num_features) // 2)
            return {
                'n_components': optuna.distributions.IntDistribution(low=2, high=upper)
            }
        elif name == 'KNN_reg':
            return {
                'n_neighbors': optuna.distributions.IntDistribution(low=1, high=50),
                'weights': optuna.distributions.CategoricalDistribution(choices=['uniform', 'distance']),
                'p': optuna.distributions.IntDistribution(low=1, high=3)
            }
        elif name == 'LightGBM_reg':
            return {
                'n_estimators': optuna.distributions.IntDistribution(low=50, high=500),
                'max_depth': optuna.distributions.IntDistribution(low=-1, high=50),
                'learning_rate': optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
                'num_leaves': optuna.distributions.IntDistribution(low=20, high=150),
                'subsample': optuna.distributions.FloatDistribution(low=0.5, high=1.0),
                'colsample_bytree': optuna.distributions.FloatDistribution(low=0.5, high=1.0)
            }
        else:
            return {}

    # Define cross-validation strategy
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def evaluate_model(name, reg, X, y, sample_ids, prefix, RANDOM_SEED):
        num_features = X.shape[1]
        # Define Optuna sampler with fixed seed
        sampler = TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        # Get hyperparameter distributions
        param_distributions = get_param_distributions(name, num_features)

        if not param_distributions:
            raise ValueError(f"No hyperparameter distributions defined for model '{name}'.")

        try:
            def objective(trial):
                # Suggest hyperparameters
                params = {}
                for param, distribution in param_distributions.items():
                    if isinstance(distribution, optuna.distributions.IntDistribution):
                        params[param] = trial.suggest_int(param, distribution.low, distribution.high)
                    elif isinstance(distribution, optuna.distributions.FloatDistribution):
                        params[param] = trial.suggest_float(param, distribution.low, distribution.high, log=distribution.log)
                    elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                        params[param] = trial.suggest_categorical(param, distribution.choices)
                    else:
                        raise ValueError(f"Unsupported distribution type: {type(distribution)}")

                # Clone model with suggested hyperparameters
                if name == 'Neural_Network_reg':
                    model = MLPRegressor(
                        hidden_layer_sizes=(params['hidden_layer_sizes'],),
                        alpha=params['alpha'],
                        learning_rate_init=params['learning_rate_init'],
                        max_iter=200000,
                        random_state=RANDOM_SEED
                    )
                elif name == 'Random_Forest_reg':
                    model = RandomForestRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        max_features=params['max_features'],
                        min_samples_split=params['min_samples_split'],
                        min_samples_leaf=params['min_samples_leaf'],
                        random_state=RANDOM_SEED,
                        n_jobs=-1
                    )
                elif name == 'SVM_reg':
                    model = SVR(
                        C=params['C'],
                        gamma=params['gamma'],
                        epsilon=params['epsilon']
                    )
                elif name == 'XGBoost_reg':
                    model = XGBRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        eval_metric='rmse',
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=-1
                    )
                elif name == 'PLS_reg':
                    model = PLSRegression(
                        n_components=params['n_components']
                    )
                elif name == 'KNN_reg':
                    model = KNeighborsRegressor(
                        n_neighbors=params['n_neighbors'],
                        weights=params['weights'],
                        p=params['p'],
                        n_jobs=-1
                    )
                elif name == 'LightGBM_reg':
                    model = LGBMRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        num_leaves=params['num_leaves'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        random_state=RANDOM_SEED,
                        force_col_wise=True,
                        verbosity=-1,
                        n_jobs=-1
                    )
                else:
                    raise ValueError(f"Unsupported model: {name}")

                # Perform cross-validation
                with SuppressOutput():
                    cv_scores = []
                    for train_idx, valid_idx in cv_outer.split(X, y):
                        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                        y_train, y_valid = y[train_idx], y[valid_idx]
                        model.fit(X_train, y_train)
                        preds = model.predict(X_valid)
                        mse = mean_squared_error(y_valid, preds)
                        cv_scores.append(mse)
                # Return average MSE
                return np.mean(cv_scores)

            # Optimize hyperparameters
            study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=False)

            # Get best hyperparameters
            best_params = study.best_params

            # Initialize model with best hyperparameters
            if name == 'Neural_Network_reg':
                best_model = MLPRegressor(
                    hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
                    alpha=best_params['alpha'],
                    learning_rate_init=best_params['learning_rate_init'],
                    max_iter=200000,
                    random_state=RANDOM_SEED
                )
            elif name == 'Random_Forest_reg':
                best_model = RandomForestRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    max_features=best_params['max_features'],
                    min_samples_split=best_params['min_samples_split'],
                    min_samples_leaf=best_params['min_samples_leaf'],
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
            elif name == 'SVM_reg':
                best_model = SVR(
                    C=best_params['C'],
                    gamma=best_params['gamma'],
                    epsilon=best_params['epsilon']
                )
            elif name == 'XGBoost_reg':
                best_model = XGBRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    learning_rate=best_params['learning_rate'],
                    subsample=best_params['subsample'],
                    colsample_bytree=best_params['colsample_bytree'],
                    eval_metric='rmse',
                    random_state=RANDOM_SEED,
                    verbosity=0,
                    n_jobs=-1
                )
            elif name == 'PLS_reg':
                best_model = PLSRegression(
                    n_components=best_params['n_components']
                )
            elif name == 'KNN_reg':
                best_model = KNeighborsRegressor(
                    n_neighbors=best_params['n_neighbors'],
                    weights=best_params['weights'],
                    p=best_params['p'],
                    n_jobs=-1
                )
            elif name == 'LightGBM_reg':
                best_model = LGBMRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    learning_rate=best_params['learning_rate'],
                    num_leaves=best_params['num_leaves'],
                    subsample=best_params['subsample'],
                    colsample_bytree=best_params['colsample_bytree'],
                    random_state=RANDOM_SEED,
                    force_col_wise=True,
                    verbosity=-1,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model: {name}")

        except Exception as e:
            print(f"Optimization failed for model {name}: {e}. Using default parameters.")
            # Initialize model with default parameters
            if name == 'Neural_Network_reg':
                best_model = MLPRegressor(
                    max_iter=200000,
                    random_state=RANDOM_SEED
                )
            elif name == 'Random_Forest_reg':
                best_model = RandomForestRegressor(
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
            elif name == 'SVM_reg':
                best_model = SVR()
            elif name == 'XGBoost_reg':
                best_model = XGBRegressor(
                    eval_metric='rmse',
                    random_state=RANDOM_SEED,
                    verbosity=0,
                    n_jobs=-1
                )
            elif name == 'PLS_reg':
                # Set n_components to 2 or floor(n_features / 2)
                n_components = 2
                upper = max(2, (num_features) // 2)
                best_model = PLSRegression(
                    n_components=n_components
                )
            elif name == 'KNN_reg':
                best_model = KNeighborsRegressor(
                    n_jobs=-1
                )
            elif name == 'LightGBM_reg':
                best_model = LGBMRegressor(
                    random_state=RANDOM_SEED,
                    force_col_wise=True,
                    verbosity=-1,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model: {name}")

        # Train the best model on the entire dataset
        with SuppressOutput():
            best_model.fit(X, y)

        # Save the best model with prefix and ' reg' appended to the model name
        model_path = f"{prefix}_{name}_best_model.pkl"
        dump(best_model, model_path)

        # Perform cross-validation predictions
        y_pred = cross_val_predict(best_model, X, y, cv=cv_outer, n_jobs=-1)

        # Calculate performance metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Save model results and data with prefix and ' reg' appended to the model name
        # Use prefix and model name in filenames
        np.save(f"{prefix}_{name}_X.npy", X.values)
        np.save(f"{prefix}_{name}_y_true.npy", y)
        np.save(f"{prefix}_{name}_y_pred.npy", y_pred)
        np.save(f"{prefix}_{name}_feature_names.npy", X.columns.values)  # Save feature names

        # Save model parameters
        with open(f"{prefix}_{name}_model_params.json", 'w') as f:
            json.dump(best_model.get_params(), f, indent=4)

        # Plot and save Actual vs Predicted with 300 DPI
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for {name}')
        plt.savefig(f"{prefix}_{name}_prediction.png", dpi=300)
        plt.close()

        # Plot and save Residuals with 300 DPI
        residuals = y - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'Residuals for {name}')
        plt.savefig(f"{prefix}_{name}_residuals.png", dpi=300)
        plt.close()

        # Return performance metrics and predictions
        return {
            'name': name,
            'model': best_model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }

    # Wrapper function to run models in parallel
    def run_model(model_name, reg, X, y, sample_ids, prefix, RANDOM_SEED):
        try:
            result = evaluate_model(model_name, reg, X, y, sample_ids, prefix, RANDOM_SEED)
            return result
        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            return None

    # Run all selected models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_model)(name, reg, X, y, sample_ids, prefix, RANDOM_SEED)
        for name, reg in selected_regressors.items()
    )

    # Filter out failed models
    results = [res for res in results if res is not None]

    # Check if any models succeeded
    if not results:
        print("All models failed to run. Please check the error logs.")
        sys.exit(1)

    # Print each model's results
    for res in results:
        print(f"Model: {res['name']}")
        print(f"MSE: {res['mse']:.4f}")
        print(f"RMSE: {res['rmse']:.4f}")
        print(f"MAE: {res['mae']:.4f}")
        print(f"R2: {res['r2']:.4f}\n")

    # Plot and save model performance metrics table with prefix and 300 DPI
    def plot_metrics_table(results, prefix):
        metrics_data = []
        for res in results:
            metrics_data.append([
                res['name'],
                round(res['mse'], 4),
                round(res['rmse'], 4),
                round(res['mae'], 4),
                round(res['r2'], 4)
            ])

        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2'])

        if metrics_df.empty:
            print("No metrics to display.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust table size
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)  # Adjust table scale

        plt.title('Model Performance Metrics')
        plt.savefig(f"{prefix}_model_performance_metrics.png", dpi=300)
        plt.close()

    # Save model performance metrics table
    plot_metrics_table(results, prefix)

    # Save all models' performance metrics to CSV with prefix
    metrics_summary = pd.DataFrame([{
        'Model': res['name'],
        'MSE': res['mse'],
        'RMSE': res['rmse'],
        'MAE': res['mae'],
        'R2': res['r2']
    } for res in results])

    metrics_summary.to_csv(f"{prefix}_model_performance_metrics.csv", index=False)
    print(f"Model performance metrics saved to {prefix}_model_performance_metrics.csv")
    print(f"Model performance table image saved to {prefix}_model_performance_metrics.png")

    # Save all models' predictions to a single CSV file with prefix
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Original Label': y
    })

    for res in results:
        predictions_df[f"{res['name']}_Predicted"] = res['y_pred']

    predictions_df.to_csv(f"{prefix}_all_model_predictions.csv", index=False)
    print(f"All model predictions saved to {prefix}_all_model_predictions.csv")

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Run regression models with Optuna hyperparameter optimization.')

    parser.add_argument('--i', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--p', type=str, required=True, help='Prefix for output files.')
    parser.add_argument('--m', type=str, nargs='+', default=[
                        'Neural_Network_reg', 'Random_Forest_reg', 'SVM_reg', 'XGBoost_reg', 'PLS_reg', 'KNN_reg', 'LightGBM_reg'],
                        help='List of models to run. Available models: NNR, RFR, SVMR, XGBR, PLSR, KNNR, LGBMR or their full names.')

    # Parse arguments
    args = parser.parse_args()

    # Define Model Map for abbreviations
    MODEL_MAP = {
        'NNR': 'Neural_Network_reg',
        'RFR': 'Random_Forest_reg',
        'SVMR': 'SVM_reg',
        'XGBR': 'XGBoost_reg',
        'PLSR': 'PLS_reg',
        'KNNR': 'KNN_reg',
        'LGBMR': 'LightGBM_reg'
    }

    # Process model names to map abbreviations to full names
    processed_models = []
    for model in args.m:
        if model in MODEL_MAP:
            processed_models.append(MODEL_MAP[model])
        elif model in MODEL_MAP.values():
            processed_models.append(model)
        else:
            print(f"Warning: Unrecognized model name '{model}'. It will be ignored.")
    
    if not processed_models:
        print("Error: No valid models specified after processing abbreviations and full names.")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    final_models = []
    for model in processed_models:
        if model not in seen:
            seen.add(model)
            final_models.append(model)

    # Run regression function
    regression(args.i, args.p, final_models)