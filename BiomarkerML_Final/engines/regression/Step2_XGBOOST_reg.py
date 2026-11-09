from engines.regression.protein_regression import _cli, train_regression_model


def xgboost_regression_nested_cv(inp, prefix=None, feature_selection_method="no"):
    return train_regression_model(inp, prefix, "XGBoost_reg", feature_selection_method)


if __name__ == "__main__":
    _cli("XGBoost_reg")
