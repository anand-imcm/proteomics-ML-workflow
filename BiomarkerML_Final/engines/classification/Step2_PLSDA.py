from engines.classification.protein_modeling import _cli, train_classification_model


def plsda_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "plsda", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("plsda", "plsda_nested_cv")
