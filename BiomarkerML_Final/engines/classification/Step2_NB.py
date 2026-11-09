from engines.classification.protein_modeling import _cli, train_classification_model


def gaussian_nb_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "gaussiannb", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("gaussiannb", "gaussian_nb_nested_cv")
