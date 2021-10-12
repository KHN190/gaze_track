from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils import *
from evaluate import *

def train(norm_X, y, params):

    # cross-validation with 5 splits
    cv = StratifiedShuffleSplit(n_splits=10, random_state=42)

    # SVM classifier
    svm = SVC(tol=1e-4)

    # parameters

    # grid search for parameters
    grid = GridSearchCV(estimator=svm, param_grid=params, cv=cv, n_jobs=-1, scoring='f1')
    grid.fit(norm_X, y)

    # print best scores
    print("The best parameters are %s with a score of %0.4f\n" %
          (grid.best_params_, grid.best_score_))

    return grid


if __name__ == '__main__':

    X, y = build_svm_data('./train/training_set.pkl', 7)

    params = {
        'kernel': ['rbf'],
        'C': [.5],
        'max_iter': [2000],
        'cache_size': [1000],
    }
    norm_X = transform_svm_df(X)

    train_X = norm_X
    train_y = y

    save_model(train_X, 'train_X.pkl')
    save_model(train_y, 'train_y.pkl')

    svm = train(train_X, train_y, params)

    evaluate(svm.predict(train_X), train_y)
    
    save_model(svm, 'svm_7.pkl')

