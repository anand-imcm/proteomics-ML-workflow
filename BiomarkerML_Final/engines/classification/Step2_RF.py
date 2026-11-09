from engines.classification.protein_modeling import _cli, train_classification_model


def random_forest_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "random_forest", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("random_forest", "random_forest_nested_cv")
