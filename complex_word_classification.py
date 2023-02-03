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


################################################# 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, true_labels = load_file(data_file)

    evaluated_labels = []
    for word in words:
        evaluated_labels.append(1)

    evaluate(evaluated_labels, true_labels)


################################################## 2.2: Word length thresholding
def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""

    # Train Data
    train_words, training_true_labels = load_file(training_file)

    best_training_thresh, training_pred_labels = find_best_length_thresh(train_words, training_true_labels)

    evaluate(training_pred_labels, training_true_labels)

    # Development Data
    development_words, dev_true_labels = load_file(development_file)

    word_len_features = get_length_features(development_words)
    dev_pred_labels = [evaluate_freq_feature(word_length, best_training_thresh) for word_length in word_len_features]

    evaluate(dev_pred_labels, dev_true_labels)

    print("Best length threshold was ", best_training_thresh)

    return best_training_thresh

def find_best_length_thresh(wordset, label_set):
    word_len_features = get_length_features(wordset)
    all_unique_lengths = set(word_len_features)

    best_fscore = 0
    best_thresh = 0
    best_thresh_labels = []

    for length in all_unique_lengths:
        predicted_labels = [evaluate_freq_feature(word_length, length) for word_length in word_len_features]

        fscore = get_fscore(predicted_labels, label_set)
        if fscore > best_fscore:
            best_fscore = fscore
            best_thresh = length
            best_thresh_labels = predicted_labels

    return best_thresh, best_thresh_labels


def evaluate_length_feature(word_length, thresh):
    return int(word_length >= thresh)


def get_length_features(words):
    return [len(word) for word in words]


############################################ 2.3: Word frequency thresholding

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
    sorted_counts = sort_all_count_values(counts)

    # Training Data
    training_words, training_true_labels = load_file(training_file)

    best_training_thresh = find_best_frequency(training_words, training_true_labels, sorted_counts, counts)

    training_pred_labels = get_frequency_predicted_labels(best_training_thresh, training_words, counts)
    evaluate(training_pred_labels, training_true_labels)

    # Development Data
    development_words, dev_true_labels = load_file(development_file)

    development_pred_labels = get_frequency_predicted_labels(best_training_thresh, development_words, counts)
    evaluate(development_pred_labels, dev_true_labels)
    print("Best frequency threshold was ", best_training_thresh)
    return best_training_thresh


def sort_all_count_values(counts):
    word_counts = set()
    for pair in counts.items():
        word_counts.add(pair[1])  # Grabs the Count within the KV pairs

    print(len(word_counts))
    return sorted(word_counts)


def find_best_frequency(words, true_labels, sorted_counts_partition, counts_dict):
    PARTITION_CONSTANT = 1500

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
    frequency_features = get_frequency_features(words, word_counts)
    predicted_labels = [evaluate_freq_feature(feature, thresh) for feature in frequency_features]

    return predicted_labels


def evaluate_freq_feature(feature, thresh):
    return int(feature < thresh)


def get_frequency_features(words, word_count_dict):
    return [word_count_dict[word] for word in words]


def get_wordset_features_n_labels(wordset_file, counts):
    wordset, wordset_true_labels = load_file(wordset_file)

    words_freq_features = get_frequency_features(wordset, counts)
    words_len_features = get_length_features(wordset)

    wordset_features = np.array([words_freq_features, words_len_features]).T

    return wordset_features, wordset_true_labels


def normalize(feature_matrix):
    return (feature_matrix - feature_matrix.mean(axis=0))/feature_matrix.std(axis=0)


def get_train_and_dev_sets(train_file, dev_file, counts):
    train_features, train_labels = get_wordset_features_n_labels(train_file,counts)

    normalized_train_features = normalize(train_features)

    dev_features, dev_labels = get_wordset_features_n_labels(dev_file, counts)
    normalized_dev_features = normalize(dev_features)

    return normalized_train_features, train_labels, normalized_dev_features, dev_labels

################################################################# 3.1: Naive Bayes
def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    train_f, train_l, dev_f, dev_l = my_classifyer_train_and_dev_sets(training_file, development_file, counts)

    clf.fit(train_f, train_l)

    train_l_predicted = clf.predict(train_f)

    evaluate(train_l_predicted, train_l)


    dev_l_predicted = clf.predict(dev_f)

    evaluate(dev_l_predicted, dev_l)


### 3.2: Logistic Regression
def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0)

    train_f, train_l, dev_f, dev_l = my_classifyer_train_and_dev_sets(training_file, development_file, counts)

    clf.fit(train_f, train_l)

    train_l_predicted = clf.predict(train_f)

    evaluate(train_l_predicted, train_l)

    dev_l_predicted = clf.predict(dev_f)

    evaluate(dev_l_predicted, dev_l)


### 3.3: Build your own classifier
def my_classifyer_train_and_dev_sets(train_file, dev_file, counts):
    train_features, train_labels = get_wordset_features_n_labels(train_file,counts)

    normalized_train_features = (train_features)

    dev_features, dev_labels = get_wordset_features_n_labels(dev_file, counts)
    normalized_dev_features = (dev_features)

    return normalized_train_features, train_labels, normalized_dev_features, dev_labels


def my_classifier(training_file, development_file, counts):

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()

    train_f, train_l, dev_f, dev_l = my_classifyer_train_and_dev_sets(training_file, development_file, counts)

    clf.fit(train_f, train_l)

    dev_l_predicted = clf.predict(dev_f)

    evaluate(dev_l_predicted, dev_l)


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