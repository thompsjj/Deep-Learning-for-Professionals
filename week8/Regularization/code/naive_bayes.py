from collections import Counter, defaultdict
import numpy as np


class NaiveBayes(object):

    def __init__(self, alpha=1):
        '''
        INPUT:
        - alpha: float, laplace smoothing constant
        '''

        self.class_totals = None
        self.class_feature_totals = None
        self.class_counts = None
        self.alpha = alpha

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''

        self.class_totals = Counter()
        self.class_feature_totals = defaultdict(Counter)

        ### YOUR CODE HERE

    def fit(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT: None
        '''

        # compute priors
        self.class_counts = Counter(y)

        # compute likelihoods
        self._compute_likelihood(X, y)

    def predict(self, X):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix

        OUTPUT:
        - predictions: numpy array
        '''

        predictions = np.zeros(X.shape[0])

        ### YOUR CODE HERE

        return predictions

    def score(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))
