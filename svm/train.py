from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from utils import *
from evaluate import *


@ignore_warnings(category=ConvergenceWarning)
def train(norm_X, y, params, scoring='f1'):

    # cross-validation with 5 splits
    cv = StratifiedShuffleSplit(n_splits=10, random_state=42)

    # SVM classifier
    svm = SVC(tol=1e-4)

    # grid search for parameters
    grid = GridSearchCV(estimator=svm,
                        param_grid=params,
                        cv=cv,
                        n_jobs=-1,
                        scoring=scoring)
    grid.fit(norm_X, y)

    # print best scores
    print("The best parameters are %s with a score of %0.4f\n" %
          (grid.best_params_, grid.best_score_))

    return grid


if __name__ == '__main__':

    X, y = build_svm_data('./train/training_set.pkl', 13)

    params = {
        'kernel': ['rbf'],
        'C': [1, 5, 10],
        'max_iter': [2000, 3000, 4000],
        'cache_size': [1000],
    }
    norm_X = transform_svm_df(X)

    train_X = norm_X
    train_y = y

    svm = train(train_X, train_y, params, 'f1')

    evaluate(svm.predict(train_X), train_y)

    save_model(svm, 'svm_13.pkl')
