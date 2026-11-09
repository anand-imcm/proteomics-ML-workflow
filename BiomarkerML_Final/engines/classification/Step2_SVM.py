from engines.classification.protein_modeling import _cli, train_classification_model


def svm_nested_cv(inp, prefix=None, feature_selection_method="no", **kwargs):
    return train_classification_model(inp, prefix, "svm", feature_selection_method, **kwargs)


if __name__ == "__main__":
    _cli("svm", "svm_nested_cv")
