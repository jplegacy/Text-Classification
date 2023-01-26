"""Movie review classification using scikit-learn.

Author: Kristina Striegnitz

Use this code to understand how to use scikit-learn's implementation of a naive Bayes
classifier and to test your implementation of evaluation.py.
"""

from nltk.corpus import movie_reviews

from sklearn.naive_bayes import GaussianNB

import random
from collections import Counter
import numpy as np

from evaluation import evaluate

def create_training_and_dev_sets():
    # Load data
    reviews = [list(movie_reviews.words(fileid))
                   for fileid in movie_reviews.fileids('pos')]
    reviews.extend([list(movie_reviews.words(fileid))
                        for fileid in movie_reviews.fileids('neg')])
    labels = [1 for file in movie_reviews.fileids('pos')]
    labels += [0 for file in movie_reviews.fileids('neg')]
    # Split into training set and development set
    dev_selection = random.sample(range(0, len(reviews)), 500)
    dev_reviews = [reviews[i] for i in dev_selection]
    training_reviews = [reviews[i] for i in range(len(reviews)) if i not in dev_selection]
    # Turn reviews into feature representations
    # features used: for each of the 2000 most common words in the training set,
    # 1 if the word appears in a review and 0 if it doesn't (i.e. binary naive Bayes)
    training_word_counts = Counter([w.lower() for review in training_reviews for w in review])
    vocab = [word_count[0] for word_count in training_word_counts.most_common(2000)]
    training_x = np.array([create_features(r, vocab) for r in training_reviews])
    dev_x = np.array([create_features(r, vocab) for r in dev_reviews])
    training_y = np.array([labels[i] for i in range(len(labels)) if i not in dev_selection])
    dev_y = np.array([labels[i] for i in dev_selection])
    return training_x, training_y, dev_x, dev_y

def create_features(words, vocab):
    word_counts = Counter(words)
    return [int(word_counts[w] > 0) for w in vocab]

if __name__ == "__main__":
    # Create training and development/test set
    training_x, training_y, dev_x, dev_y = create_training_and_dev_sets()
    # Train scikit-learn naive Bayes classifier
    clf = GaussianNB()
    clf.fit(training_x, training_y)
    # Evaluate on dev set
    dev_y_predicted = clf.predict(dev_x)
    # For now, just print out the predicted and actual labels:
    for i in range(len(dev_y)):
        print("predicted:", dev_y_predicted[i], " actual:", dev_y[i])
    # After you have completed the code in evaluation.py, you can uncomment the
    # following line to get precision, recall, and f-score for the dev set. The
    # f-measure should be around 80%.
    # evaluate(dev_y_predicted, dev_y)
