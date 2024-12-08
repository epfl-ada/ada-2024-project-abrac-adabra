# model_training.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

def train_knn(X, y,cv=5):
    # k-NN: cross-validation on the number of neighbors
    knn_clf = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 100)}
    knn_gscv = GridSearchCV(knn_clf, param_grid, cv=cv, scoring='balanced_accuracy')
    knn_gscv.fit(X, y)
    print(f'k-NN: best balanced accuracy of {knn_gscv.best_score_:.4f}, for {knn_gscv.best_params_}.')
    return knn_gscv

def train_svm(X, y, cv=5):
    # SVM with RBF as default kernel
    svm_clf = SVC(random_state=17)
    score = cross_val_score(svm_clf, X, y, cv=cv, scoring='balanced_accuracy').mean()
    print(f'SVM: CV balanced accuracy of {score:.4f}.')
    return svm_clf

'''def train_logistic(X,y, cv = 5):
    # Multinomial logistic regression, loss: choose between OvR and cross-entropy
    ll_clf = LogisticRegression(random_state=17)
    param_grid = {'multi_class': ['multinomial', 'ovr']}
    ll_gscv = GridSearchCV(ll_clf, param_grid, cv=cv, scoring='balanced_accuracy')
    ll_gscv.fit(X, y);
    print(f'Logistic Regression: best balanced accuracy of {ll_gscv.best_score_:.4f}, for the loss: {ll_gscv.best_params_}.')
    return ll_gscv'''


def train_logistic(X, y, cv=5):
    # Suppress specific warnings
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    
    # Multinomial logistic regression, loss: choose between OvR and cross-entropy
    ll_clf = LogisticRegression(max_iter=500, random_state=17)  # Increase max_iter for convergence
    param_grid = {'multi_class': ['multinomial', 'ovr']}
    ll_gscv = GridSearchCV(ll_clf, param_grid, cv=cv, scoring='balanced_accuracy')
    ll_gscv.fit(X, y)
    
    print(f'Logistic Regression: best balanced accuracy of {ll_gscv.best_score_:.4f}, for the loss: {ll_gscv.best_params_}.')
    return ll_gscv


def train_mlp(X,y,num_layers = 5, cv = 10):
    if num_layers == 1:
        mlp_clf = MLPClassifier(random_state=17)
        param_grid = {'hidden_layer_sizes': [(i, ) for i in np.arange(10, 100, 20)]}
        mlp_gscv = GridSearchCV(mlp_clf, param_grid, cv=cv, scoring='balanced_accuracy')
        mlp_gscv.fit(X, y);
        print(f'MLP with 1 hidden layer: best balanced accuracy of {mlp_gscv.best_score_:.4f}, for a number of neurons per layer: {mlp_gscv.best_params_}.')
        return mlp_gscv

    elif num_layers == 2:
   
        # MLP with 2 hidden layers: CV on the number of neurons per layer
        mlp_clf = MLPClassifier(random_state=17)
        param_grid = {'hidden_layer_sizes': [(i, i) for i in np.arange(10, 100, 20)]}
        mlp_gscv = GridSearchCV(mlp_clf, param_grid, cv=5, scoring='balanced_accuracy')
        mlp_gscv.fit(X, y);
        print(f'MLP with 2 hidden layers: best balanced accuracy of {mlp_gscv.best_score_:.4f}, for a number of neurons per layer: {mlp_gscv.best_params_}.')
        return mlp_gscv
    
    elif num_layers == 3:
        # MLP with 3 hidden layers: CV on the number of neurons per layer
        mlp_clf = MLPClassifier(random_state=17)
        param_grid = {'hidden_layer_sizes': [(i, i, i) for i in np.arange(10, 100, 20)]}
        mlp_gscv = GridSearchCV(mlp_clf, param_grid, cv=5, scoring='balanced_accuracy')
        mlp_gscv.fit(X, y);
        print(f'MLP with 3 hidden layers: best balanced accuracy of {mlp_gscv.best_score_:.4f}, for a number of neurons per layer: {mlp_gscv.best_params_}.')
        return mlp_gscv

    elif num_layers == 5:
        mlp_clf = MLPClassifier(random_state=17, max_iter=500, hidden_layer_sizes=(200, 200, 100, 100, 50));
        print(f'MLP with 5 hidden layers: balanced accuracy of {cross_val_score(mlp_clf, X, y, cv=cv, scoring="balanced_accuracy").mean():.4f}.')
        return mlp_clf