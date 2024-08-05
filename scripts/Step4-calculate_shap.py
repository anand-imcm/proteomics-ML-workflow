import pandas as pd
import numpy as np
import argparse
import joblib
import shap
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import warnings

# 抑制所有警告信息
warnings.filterwarnings('ignore')

def calculate_shap_values_nn(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[train_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 10))
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def calculate_shap_values_rf(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def calculate_shap_values_knn(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 10))
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def calculate_shap_values_plsda(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def calculate_shap_values_svm(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 10))
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def calculate_shap_values_xgb(best_model, X, y_encoded, cv_outer, num_classes):
    shap_values_list = []
    for train_idx, test_idx in cv_outer.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        if num_classes == 2:
            shap_values = np.array(shap_values).reshape(-1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_values = np.array(shap_values).reshape(num_classes, -1, X.shape[1])
            shap_values = np.mean(np.abs(shap_values), axis=1).mean(axis=0)

        shap_values_list.append(shap_values)

    shap_values_mean = np.mean(shap_values_list, axis=0)
    return shap_values_mean

def load_model_and_data(model_name):
    if model_name == 'neural_network':
        model_path = 'neural_network_model.pkl'
        data_path = 'neural_network_data.pkl'
    elif model_name == 'random_forest':
        model_path = 'random_forest_model.pkl'
        data_path = 'random_forest_data.pkl'
    elif model_name == 'knn':
        model_path = 'knn_model.pkl'
        data_path = 'knn_data.pkl'
    elif model_name == 'plsda':
        model_path = 'plsda_model.pkl'
        data_path = 'plsda_data.pkl'
    elif model_name == 'svm':
        model_path = 'svm_model.pkl'
        data_path = 'svm_data.pkl'
    elif model_name == 'xgboost':
        model_path = 'xgboost_model.pkl'
        data_path = 'xgboost_data.pkl'
    elif model_name == 'vae':
        return None, pd.DataFrame(), np.array([]), 0
    else:
        raise ValueError("Model name not recognized.")

    best_model = joblib.load(model_path)
    X, y_encoded, le = joblib.load(data_path)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    num_classes = len(np.unique(y_encoded))
    return best_model, X, y_encoded, num_classes

def plot_shap_radar(model, shap_values_mean, feature_names, num_features):
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': shap_values_mean
    })
    shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False).head(num_features)

    labels = shap_df['Feature'].values
    values = shap_df['Mean SHAP Value'].values
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    plt.title(f'SHAP Values for {model}', size=20, color='black', weight='bold')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate SHAP values for specified models')
    parser.add_argument('--models', type=str, nargs='+', choices=['neural_network', 'random_forest', 'knn', 'plsda', 'svm', 'xgboost', 'vae'], required=True, help='Model names: neural_network, random_forest, knn, plsda, svm, xgboost, vae')
    parser.add_argument('--num_features', type=int, required=True, help='Number of top features to display in radar chart')
    args = parser.parse_args()

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    for model_name in args.models:
        if model_name == 'vae':
            shap_df = pd.read_csv('vae_shap_values.csv')
            feature_names = shap_df['Feature']
            shap_values_mean = shap_df['Mean SHAP Value']
        else:
            best_model, X, y_encoded, num_classes = load_model_and_data(model_name)
            if model_name == 'neural_network':
                shap_values_mean = calculate_shap_values_nn(best_model, X, y_encoded, cv_outer, num_classes)
            elif model_name == 'random_forest':
                shap_values_mean = calculate_shap_values_rf(best_model, X, y_encoded, cv_outer, num_classes)
            elif model_name == 'knn':
                shap_values_mean = calculate_shap_values_knn(best_model, X, y_encoded, cv_outer, num_classes)
            elif model_name == 'plsda':
                shap_values_mean = calculate_shap_values_plsda(best_model, X, y_encoded, cv_outer, num_classes)
            elif model_name == 'svm':
                shap_values_mean = calculate_shap_values_svm(best_model, X, y_encoded, cv_outer, num_classes)
            elif model_name == 'xgboost':
                shap_values_mean = calculate_shap_values_xgb(best_model, X, y_encoded, cv_outer, num_classes)

        # 保存SHAP值
        if model_name != 'vae':
            feature_names = X.columns
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean SHAP Value': shap_values_mean
            })
            # 获取脚本所在路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            shap_df.to_csv(os.path.join(script_dir, f'{model_name}_shap_values.csv'), index=False)

            print(f'SHAP values calculated and saved to {model_name}_shap_values.csv')

        # 绘制雷达图
        plot_shap_radar(model_name, shap_values_mean, feature_names, args.num_features)
