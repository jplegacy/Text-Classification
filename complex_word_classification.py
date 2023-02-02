"""Text classification for identifying complex words.

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate


def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, true_labels = load_file(data_file)

    evaluated_labels = []
    for word in words:
        evaluated_labels.append(1)

    evaluate(evaluated_labels, true_labels)


### 2.2: Word length thresholding
def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    # Train Data
    train_words, training_true_labels = load_file(training_file)

    best_thresh, training_pred_labels = find_best_length_thresh(train_words, training_true_labels)

    evaluate(training_pred_labels, training_true_labels)

    # Development Data
    development_words, dev_true_labels = load_file(development_file)

    development_pred_labels = []
    for word in development_words:
        development_pred_labels.append(int(len(word) > best_thresh))
    evaluate(development_pred_labels, dev_true_labels)

    return best_thresh, training_pred_labels, development_pred_labels


def find_best_length_thresh(wordset, label_set):
    word_sized_labels = {}

    best_fscore = 0
    best_thresh = 0

    # Loop creates data structure to store each word-size and there respective labels
    for word in wordset:
        if len(word) not in word_sized_labels.keys():
            word_sized_labels[len(word)] = []

    for word_size in word_sized_labels.keys():
        for word in wordset:
            length_condition = len(word) > word_size
            word_sized_labels[word_size].append(int(length_condition))

        predicted_labels = word_sized_labels[word_size]
        fscore = get_fscore(predicted_labels, label_set)
        if fscore > best_fscore:
            best_fscore = fscore
            best_thresh = word_size
    return best_thresh, word_sized_labels[best_thresh]


### 2.3: Word frequen  cy thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """

    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt', encoding="utf8") as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts


def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """

    words, training_true_labels = load_file(training_file)
    sorted_counts = sorted_word_counts(counts)

    best_thresh = find_best_frequency(words, training_true_labels, sorted_counts, counts)
    print("Best Thresh for x ", best_thresh )
    training_pred_labels = get_frequency_predicted_labels(best_thresh, words, counts)

    # best_thresh = 44
    # training_pred_labels = get_frequency_predicted_labels(best_thresh, words, counts)

    evaluate(training_pred_labels, training_true_labels)

    # Development Data
    development_words, dev_true_labels = load_file(development_file)
    development_pred_labels = get_frequency_predicted_labels(best_thresh, development_words, counts)
    evaluate(development_pred_labels, dev_true_labels)

    return best_thresh, training_pred_labels, development_pred_labels


def sorted_word_counts(counts):
    word_counts = set()
    for pair in counts.items():
        word_counts.add(pair[1])  # Grabs the Count within the KV pairs

    return sorted(word_counts)


def find_best_frequency(words, true_labels, sorted_counts_partition, counts_dict):
    PARTITION_CONSTANT = 5

    # Base Case
    if len(sorted_counts_partition) <= PARTITION_CONSTANT:
        best_thresh = get_middle_value(0, len(sorted_counts_partition) - 1, sorted_counts_partition)

        return best_thresh

    partition_step_size = int(len(sorted_counts_partition) / PARTITION_CONSTANT)

    # Gets partition intervals to split up sorted counts list
    partition_intervals = []
    for partition_itr in range(0, PARTITION_CONSTANT):
        starting_index = (partition_itr * partition_step_size)
        ending_index = starting_index + partition_step_size
        partition_intervals.append((starting_index, ending_index))

    # Checks each intervals middle element and finds the one with the highest fscore
    best_interval = partition_intervals[0]
    best_fscore = 0
    for i, interval in enumerate(partition_intervals):
        middle_frequency = get_middle_value(interval[0], interval[1], sorted_counts_partition)
        predicted_labels = get_frequency_predicted_labels(middle_frequency, words, counts_dict)

        fscore = get_fscore(predicted_labels, true_labels)
        if fscore > best_fscore:
            best_fscore = fscore
            best_interval = partition_intervals[i]

    sub_sorted_count_list = sorted_counts_partition[best_interval[0]:best_interval[1]]

    return find_best_frequency(words, true_labels, sub_sorted_count_list, counts_dict)


def get_middle_value(starting, ending, list_intervals):
    return list_intervals[int((starting + ending) / 2)]


def get_frequency_predicted_labels(thresh, words, word_counts):
    predicted_labels = []
    for word in words:
        if word_counts[word] is None:
            predicted_labels.append(0)
            continue
        if thresh is None:
            predicted_labels.append(0)
            continue

        frequency_condition = word_counts[word] < thresh
        predicted_labels.append(int(frequency_condition))

    return predicted_labels


### 3.1: Naive Bayes
def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    training_words, training_true_labels = load_file(training_file)
    development_words, development_true_labels = load_file(development_file)


    # clf.fit(training_x, training_y)




### 3.2: Logistic Regression
def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    ## YOUR CODE HERE
    pass


### 3.3: Build your own classifier

def my_classifier(training_file, development_file, counts):
    ## YOUR CODE HERE
    pass


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)


def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)


if __name__ == "__main__":
    training_file = "/var/csc483/data/complex_words_training.txt"
    development_file = "/var/csc483/data/complex_words_development.txt"
    test_file = "/var/csc483/data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "/var/csc483/ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE
