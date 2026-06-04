from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

def train_svc(X, y, cv_folds=5):
    model = SVC(kernel='linear', random_state=7)

    scores = cross_validate(
        estimator = model,
        X = X,
        y = y,
        cv = cv_folds,
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score = False
    )

    return scores