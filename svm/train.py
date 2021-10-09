from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils import *
from utils_frame_based import *


def train(norm_X, y, params):

    # cross-validation with 5 splits
    cv = StratifiedShuffleSplit(n_splits=10, random_state=42)

    # SVM classifier
    svm = SVC(tol=1e-5)

    # parameters

    # grid search for parameters
    grid = GridSearchCV(estimator=svm, param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(norm_X, y)

    # print best scores
    print("The best parameters are %s with a score of %0.4f\n" %
          (grid.best_params_, grid.best_score_))

    return grid


if __name__ == '__main__':

    X, y = build_svm_data('./train/training_set.pkl', 7)

    norm_X = transform_svm_data(X)

    params = {
        'kernel': ['rbf'],
        'C': [10],
        'gamma': ['scale'],
        'max_iter': [5000],
        'class_weight': [None],
    }
    svm = train(norm_X, y, params)

    # svm.predict(norm_X[0].reshape(1, -1))

    save_model(svm, 'svm_7.pkl')
