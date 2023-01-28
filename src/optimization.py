from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from transformers import MultipleScatterCorrection, SavitzkyGolay
from transformers import StandardNormalVariate

def model_optimization(X, y):

    # Define a Standard Scaler to normalize inputs
    scaler = MultipleScatterCorrection()
    derivative = SavitzkyGolay()

# set the tolerance to a large value to make the example faster
    rf = RandomForestClassifier()

    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {'rf__bootstrap': [True],
                  'rf__max_depth': [80, 90, 100, 110],
                  'rf__max_features': [2, 3],
                  'rf__min_samples_leaf': [3, 4, 5],
                  'rf__min_samples_split': [8, 10, 12],
                  'rf__max_depth': [80, 90, 100, 110],
                  'rf__n_estimators': [100, 200, 300, 1000]}

    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X, y.values.ravel())
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

def model_optimization_LR(X, y):

    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    #derivative = SavitzkyGolay(filter_win=11, deriv_order=0) for age
    derivative = SavitzkyGolay(filter_win=9, deriv_order=2)

# set the tolerance to a large value to make the example faster
    rf = LogisticRegression(max_iter=10000, random_state=123)

    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {'rf__penalty': ['l1', 'l2'],
                  'rf__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'rf__class_weight': [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
                  'rf__solver': ['liblinear', 'saga']}

    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X, y.ravel())
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)


def model_optimization_SVC(X, y):

    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    derivative = SavitzkyGolay(filter_win=11, deriv_order=1)

# set the tolerance to a large value to make the example faster
    rf = SVC(class_weight='balanced')

    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {'rf__kernel': ['rbf', 'poly', 'sigmoid'],
                  'rf__C': [0.1, 1, 10, 100],
                  'rf__gamma': [1, 0.1, 0.01, 0.001]}

    search = GridSearchCV(pipe, param_grid, n_jobs=2,scoring='recall')
    search.fit(X, y.ravel())
    best_model = search.best_params_
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    return search