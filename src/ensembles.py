import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import random


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None, random_state=5,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.random_state = random_state
        self.trees_parameters = trees_parameters
        np.random.seed(self.random_state)
        # Для получения информации о значениях функции потерь после каждой итерации
        self.train_loss_functions = []
        self.val_loss_functions = []

    def fit(self, X, y, validation, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        
        self.algorithms = []
        for i in range(self.n_estimators):
            inds = []
            for j in range(X.shape[0]):
                inds.append(random.randint(0, X.shape[0] - 1))
            features = 0
            if not self.feature_subsample_size:
                features = round(X.shape[1] / 3)
            else:
                features = self.feature_subsample_size
            # Обучение не на подвыборке признаков (это плохо, поскольку часть базовых моделей может обучаться на плохих признаках ->
            # высокое смещение). Лучший предикат в каждой вершине выбираем из случайного подмножества признаков (своё подмножество в каждой
            # вершине!)
            model = DecisionTreeRegressor(splitter='random', max_depth=self.max_depth, max_features=features,
                                          random_state=self.random_state, **self.trees_parameters)
            model.fit(X[inds], y[inds])
            self.algorithms.append(model)
            self.train_loss_functions.append(np.mean((y - self.predict(X)) ** 2))
            if validation:
                self.val_loss_functions.append(np.mean((y_val - self.predict(X_val)) ** 2))
        return self.train_loss_functions, self.val_loss_functions

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answers = []
        for model in self.algorithms:
            answers.append(model.predict(X))
        return np.mean(answers, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None, random_state=5,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.random_state = random_state
        self.trees_parameters = trees_parameters
        np.random.seed(self.random_state)
        # Для получения информации о значениях функции потерь после каждой итерации
        self.train_loss_functions = []
        self.val_loss_functions = []


    def fit(self, X, y, validation, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.algorithms = []
        self.weights = []
        self.preds = []
        self.y = y.copy()
        for i in range(self.n_estimators):
            features = 0
            if not self.feature_subsample_size:
                features = round(X.shape[1] / 3)
            else:
                features = self.feature_subsample_size
            model = DecisionTreeRegressor(splitter='random',
                                          max_depth=self.max_depth,
                                          max_features=features,
                                          random_state=self.random_state,
                                          **self.trees_parameters)
            model.fit(X, self.y)
            self.algorithms.append(model)
            self.pred = model.predict(X)
            self.preds.append(self.pred)
            if self.weights == []:
                a = self.preds[0] * 1 * self.learning_rate
            else:
                a = np.sum(
                    [
                        self.weights[i] * self.preds[i]
                        for i in range(len(self.weights))
                    ],
                    axis=0,
                )
            
            self.weights.append(minimize_scalar(lambda xp: (np.sum((a + xp * self.pred - y) ** 2))).x)
            if i == 0:
                self.weights[0] = 1
            self.y = self.y - self.weights[-1] * self.learning_rate * self.pred

            self.train_loss_functions.append(np.mean((y - self.predict(X)) ** 2))
            if validation:
                self.val_loss_functions.append(np.mean((y_val - self.predict(X_val)) ** 2))
        return self.train_loss_functions, self.val_loss_functions
    



    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answers = []
        for i, model in enumerate(self.algorithms):
            answers.append(model.predict(X) * self.weights[i])
        print(answers)
        return np.sum(answers, axis=0)

