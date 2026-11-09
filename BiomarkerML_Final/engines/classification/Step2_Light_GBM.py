from engines.classification.protein_modeling import _cli, train_classification_model


def lightgbm_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "lightgbm", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("lightgbm", "lightgbm_nested_cv")
