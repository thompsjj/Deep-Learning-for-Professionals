from naive_bayes import NaiveBayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
import numpy as np


if __name__ == '__main__':
    data = np.genfromtxt('data/spam.csv', delimiter=',')

    y = data[:, -1]
    X = data[:, 0:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print 'Train shape:', X_train.shape
    print 'Test shape:', X_test.shape

    print

    print "My Implementation:"
    my_nb = NaiveBayes()
    my_nb.fit(X_train, y_train)
    print 'Accuracy:', my_nb.score(X_test, y_test)
    my_predictions =  my_nb.predict(X_test)

    print

    print "sklearn's Implementation"
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    print 'Accuracy:', mnb.score(X_test, y_test)
    sklearn_predictions = mnb.predict(X_test)

    # Assert I get the same results as sklearn
    # (will give an error if different)
    assert np.all(sklearn_predictions == my_predictions)
