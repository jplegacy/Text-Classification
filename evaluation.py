"""Evaluation Metrics

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_counts = 0

    for index, label in enumerate(y_pred):
        corresponding_label = y_true[index]
        if label == corresponding_label:
            true_counts += 1

    total_labels = len(y_pred)

    if total_labels == 0:
        return 0

    return true_counts/total_labels


def get_precision(y_pred, y_true):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """

    true_positives = 0
    false_positives = 0

    for index, label in enumerate(y_pred):
        corresponding_true_label = y_true[index]

        if corresponding_true_label and label:
            true_positives += 1
            continue
        elif not corresponding_true_label and label:
            false_positives += 1
            continue

    if true_positives+false_positives == 0:
        return 0

    return true_positives / (true_positives+false_positives)


def get_recall(y_pred, y_true):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    true_positives = 0
    false_negatives = 0

    for index, label in enumerate(y_pred):
        corresponding_true_label = y_true[index]

        if corresponding_true_label and label:
            true_positives += 1
            continue
        elif corresponding_true_label and not label:
            false_negatives += 1
            continue

    if true_positives + false_negatives == 0:
        return 0

    return true_positives / (true_positives + false_negatives)


def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    precision = get_precision(y_pred, y_true)

    recall = get_recall(y_pred, y_true)

    if precision+recall == 0:
        return 0
    return 2*precision*recall / (precision+recall)


def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    scores = [("Accuracy",get_accuracy(y_pred,y_true)),
              ("Precision",get_precision(y_pred,y_true)),
              ("Recall",get_recall(y_pred,y_true)),
              ("Fscore",get_fscore(y_pred,y_true))]

    for metric in scores:
        print(metric[0] + ": " + str(metric[1] * 100) + "%")

