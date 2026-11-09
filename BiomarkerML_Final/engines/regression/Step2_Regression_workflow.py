import argparse
from pathlib import Path

import pandas as pd

from engines.regression.protein_regression import train_regression_model
from core.publication_utils import (
    DEFAULT_RANDOM_STATE,
    configure_publication_plots,
    set_global_seed,
    task_run_context,
    resolve_model_feature_selection,
)

MODEL_ALIASES = {
    "NN_reg": "Neural_Network_reg",
    "Neural_Network_reg": "Neural_Network_reg",
    "RF_reg": "Random_Forest_reg",
    "Random_Forest_reg": "Random_Forest_reg",
    "SVM_reg": "SVM_reg",
    "XGB_reg": "XGBoost_reg",
    "XGBoost_reg": "XGBoost_reg",
    "PLS_reg": "PLS_reg",
    "KNN_reg": "KNN_reg",
    "LGBM_reg": "LightGBM_reg",
    "LightGBM_reg": "LightGBM_reg",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiomarkerML regression models.")
    parser.add_argument("--input", "-i", "--csv", dest="csv", required=True)
    parser.add_argument("-p", "--prefix", default=None)
    parser.add_argument("-m", "--models", nargs="+", required=True, choices=sorted(MODEL_ALIASES))
    parser.add_argument("-f", "--feature_selection", default="no", help="yes/no automatic protein feature selection")
    parser.add_argument("--outdir", default=".", help="Base output directory")
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def combine_summary(prefix: str, models: list[str]) -> None:
    rows = []
    for model in models:
        path = f"{prefix}_{model}_model_performance_metrics.csv"
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        rows.append(
            {
                "Model": model,
                "MSE": pd.to_numeric(df.get("MSE"), errors="coerce").mean(),
                "MAE": pd.to_numeric(df.get("MAE"), errors="coerce").mean(),
                "RMSE": pd.to_numeric(df.get("RMSE"), errors="coerce").mean(),
                "R2": pd.to_numeric(df.get("R2"), errors="coerce").mean(),
            }
        )
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{prefix}_models_summary.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_df = summary.set_index("Model")[["RMSE", "MAE"]]
        plot_df.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Error")
        ax.set_title("Regression Model Summary")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{prefix}_models_summary_bar_chart.png")
        plt.close(fig)
    except Exception:
        pass


def main():
    configure_publication_plots()
    args = parse_args()
    set_global_seed(args.seed)
    prefix = args.prefix or Path(args.csv).stem
    resolved_models = [MODEL_ALIASES[m] for m in args.models]
    with task_run_context(args.outdir, "regression"):
        for model in resolved_models:
            feature_selection = resolve_model_feature_selection(args.feature_selection, model, "regression")
            train_regression_model(args.csv, prefix, model, feature_selection)
        combine_summary(prefix, resolved_models)


if __name__ == "__main__":
    main()
