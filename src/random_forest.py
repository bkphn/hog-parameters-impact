from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold

def train(X, y, groups, n_estimators = 100, cv_folds = 5, class_weight=None):
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
        groups = groups,
        cv = GroupKFold(n_splits=cv_folds),
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score = False
    )

    return scores