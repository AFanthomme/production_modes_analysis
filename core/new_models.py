import logging
from operator import itemgetter
import matplotlib.pyplot as p
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def base_change(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n /= b
    return digits


def n_dim_iterator(dimension, support=(0, 1), n_points=10):
    """
    Basic iterator on a n-dimensional square grid of given support with a given number of points in each direction.
    
    :param dimension: 
    :param support: 
    :param n_points: 
    :return: list of length dimension with the position of the current point on the grid
    """
    plop = np.linspace(support[0], support[1], n_points)
    for flat_idx in range(n_points**dimension):
        coordinates = base_change(flat_idx, n_points)
        while len(coordinates) < dimension:
            coordinates.append(0)
        yield [plop[i] for i in coordinates]


class SelfThresholdingAdaClassifier:
    def __init__(self, base_estimator, n_estimators=50, n_points=3):
        self.UncalibratedAdaBoost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
        self.CalibratedAdaBoost = self.UncalibratedAdaBoost
        self.n_points = n_points
        self.n_classes_ = None
        self.is_fitted = False
        self.is_optimized = False
        self.ground_truth = None
        self.thresholds = []
        self.history = {}
        self.scores = None
        self.y_thr = None
        self.X_thr = None


    def filter_scores(self):
        # If an event has category 0 score above background_threshold, goes to category 0
        background_threshold = 1. - self. thresholds[0]
        self.thresholds[0] = 0
        background_like = np.where(self.scores[:, 0] > background_threshold)
        self.scores[background_like, 0] = 1.01

        # Rectify scores : if score for category i is lower than thresholds[i], set it to 0
        scores_filter = ((self.scores - self.thresholds) >= 0).astype(int)
        filtered_scores = np.multiply(self.scores, scores_filter)
        return filtered_scores


    def __confusion_score(self, predictions, ground_truth, weights=None):
        # [ [false_positive, false_negative] for cat in categories]
        if not weights:
            weights = np.ones((self.n_classes_, 2))
            weights[0, 0] = 2
            weights[0, 1] = 4

        confusion_matrix = np.zeros((self.n_classes_, 2))
        mask = predictions != ground_truth

        misclassified_from = ground_truth[mask]
        misclassified_into = predictions[mask]

        for cat in range(self.n_classes_):
            confusion_matrix[cat, 0] = np.sum(misclassified_into == cat)
            confusion_matrix[cat, 1] = np.sum(misclassified_from == cat)

        confusion_matrix /= len(predictions)
        return np.sum(np.multiply(confusion_matrix, weights))

    def fit(self, X, y, weights=None):
        """
        Split the training dataset in three : base training, calibration, thresh optimization
        The thresh optimization ones are used to get the scores / ground truth for optimize_threshold.
        :param X: 
        :param y: 
        :return: 
        """

        if np.any(weights):
            X_train, X_cal, y_train, y_cal, w_train, w_cal = train_test_split(X, y, weights, test_size=0.66)
            self.UncalibratedAdaBoost.fit(X_train, y_train, w_train)
        else:
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.66)
            self.UncalibratedAdaBoost.fit(X_train, y_train)
        X_thr, X_cal, y_thr, y_cal = train_test_split(X_cal, y_cal, test_size=0.5)

        self.n_classes_ = self.UncalibratedAdaBoost.n_classes_
        self.thresholds = [0 for _ in range(self.n_classes_)]
        logging.info('\tRaw adaboosted model trained')
        self.CalibratedAdaBoost = CalibratedClassifierCV(self.UncalibratedAdaBoost, cv="prefit", method="sigmoid")
        self.CalibratedAdaBoost = self.CalibratedAdaBoost.fit(X_cal, y_cal)

        logging.info('\tModel calibrated')
        self.is_fitted = True
        self.scores = self.CalibratedAdaBoost.predict_proba(X_thr)
        self.ground_truth = y_thr
        return self

    def predict_proba(self, X_test):
        if not self.is_fitted:
            logging.error('Trying to predict probas with an unfitted base model')
            raise AttributeError
        proba = self.CalibratedAdaBoost.predict_proba(X_test)
        logging.info((np.mean(proba, axis=0), np.min(proba, axis=0), np.max(proba, axis=0)))
        self.scores = proba
        return proba

    def predict(self, X_test):
        if not self.is_optimized:
            logging.warning('Predicting without thresholding')
        self.scores = self.CalibratedAdaBoost.predict_proba(X_test)
        filtered_scores = self.filter_scores()
        return self.CalibratedAdaBoost.classes_[np.argmax(filtered_scores, axis=1)]


    def __assess_thresholds(self, thresholds):
        self.thresholds = thresholds
        predictions = self.CalibratedAdaBoost.classes_[np.argmax(self.filter_scores(), axis=1)]
        confusion_score = self.__confusion_score(predictions, self.ground_truth)
        self.history[tuple(thresholds)] = confusion_score
        return self


    def explore_thresholds(self, limits=(0., 0.5), n_points=None):
        if not n_points:
            n_points = self.n_points
        for thresholds in n_dim_iterator(self.n_classes_, support=limits, n_points=n_points):
            self.__assess_thresholds(thresholds)

    def explore_history(self):
        sorted_history = sorted(self.history.items(), key=itemgetter(1))
        weights = [pair[0] for pair in sorted_history]
        scores = [pair[1] for pair in sorted_history]
        logging.warn('Optimal weights : ' + str(weights[0]))
        mean, std = np.mean(scores), np.std(scores)
        significances = (scores - mean) / std
        p.hist(significances)
        p.title('Mean : ' + str(mean) + ' , Std : ' + str(std))
        self.thresholds = list(weights[0])
        self.is_optimized = True
 
    def score(self, X_test, truth):
        return np.mean(self.predict(X_test) == truth)
