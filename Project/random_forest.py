import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

def train(X, y, n_estimators = 100, cv_folds = 5, class_weight=None):
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        n_jobs = 5,
        class_weight = class_weight,
        random_state = 7
    )

    scores = cross_validate(
        estimator = model,
        X = X,
        y = y,
        cv = cv_folds,
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score = False
    )

    return scores