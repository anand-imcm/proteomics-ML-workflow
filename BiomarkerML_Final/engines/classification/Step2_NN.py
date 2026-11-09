from engines.classification.protein_modeling import _cli, train_classification_model


def neural_network_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "neural_network", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("neural_network", "neural_network_nested_cv")
