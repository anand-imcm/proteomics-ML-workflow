from engines.classification.protein_modeling import _cli, train_classification_model


def xgboost_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "xgboost", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("xgboost", "xgboost_nested_cv")
