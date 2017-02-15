from math import log


def CalcShannonEnt(data_set):
    """ Calculate the Shannon Entropy.

    Arguments:
        data_set: The object dataset.

    Returns:
        shannon_ent: The Shannon entropy of the object data set.
    """

    # Initiation
    num_entries = len(data_set)
    label_counts = {}
    # Statistics the frequency of each class in the dataset
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # Calculates the Shannon entropy
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent
