import numpy as np

from twenty_q.readouts import fit_logreg, loo_accuracy_logreg


def test_fit_logreg_predicts_training_labels_on_separable_data():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [2.0, 2.0],
            [1.5, 1.0],
            [2.0, -2.0],
            [1.2, -1.5],
        ]
    )
    y = ["cat", "cat", "dog", "dog", "horse", "horse"]

    dec = fit_logreg(X, y)

    assert dec.predict(X) == y


def test_loo_accuracy_logreg_is_perfect_on_separable_three_class_data():
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [2.0, 2.0],
            [1.5, 1.0],
            [2.0, -2.0],
            [1.2, -1.5],
        ]
    )
    y = ["cat", "cat", "dog", "dog", "horse", "horse"]

    acc = loo_accuracy_logreg(X, y, ["cat", "dog", "horse"])

    assert acc == 1.0
