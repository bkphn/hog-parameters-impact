from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GroupKFold

def train_svc(X, y, groups, cv_folds=5):
    model = SVC(kernel='linear', random_state=7)

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