from math import log
import operator


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


def CreateDataSet():
    """ A naive data generation method.

    Returns:
        data_set: The data set excepts label info.
        labels: The data set only contains label info.
    """

    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def SplitDataSet(data_set, axis, value):
    """ Split the data set according to the given axis and correspond value.

    Arguments:
        data_set: Object data set.
        axis: The split-feature index.
        value: The value of the split-feature.

    Returns:
        ret_data_set: The splited data set.
    """

    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def ChooseBestFeatureToSplit(data_set):
    """ Choose the best feature to split.

    Arguments:
        data_set: Object data set.

    Returns:
        best_feature: The index of the feature to split.
    """

    # Initiation
    # Because the range() method excepts the lastest number
    num_features = len(data_set[0]) - 1
    base_entropy = CalcShannonEnt(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # Choose the i-th feature of all data
        feat_list = [example[i] for example in data_set]
        # Abandon the repeat feature value(s)
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # Calculates the Shannon entropy of the splited data set
        for value in unique_vals:
            sub_data_set = SplitDataSet(data_set, i, value)
            prob = len(sub_data_set) / len(data_set)
            new_entropy += prob * CalcShannonEnt(sub_data_set)
        # base_entropy is equal or greatter than new_entropy
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    """ Decided the final class.

    When the splited data is not belongs to the same class
    while all feature is handled, the final class is decided by
    the majority class.

    Arguments:
        class_list: The class list of the splited data set.

    Returns:
        sorted_class_count[0][0]: The majority class.
    """

    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(
        class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def CreateTree(data_set, labels):
    """ Create decision tree.

    Arguments:
        data_set: The object data set.
        labels: The feature labels in the data_set.

    Returns:
        my_tree: A dict that represents the decision tree.
    """

    class_list = [example[-1] for example in data_set]
    # If the classes are fully same
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # If all feature is handled
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # Get the best split-feature and the correspond label
    best_feat = ChooseBestFeatureToSplit(data_set)
    best_feat_label = labels[best_feat]
    # Build a recurrence dict
    my_tree = {best_feat_label: {}}
    # Get the next step labels parameter
    del(labels[best_feat])
    # Next step start
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        # Recurrence calls
        my_tree[best_feat_label][value] = CreateTree(
            SplitDataSet(data_set, best_feat, value), sub_labels)

    return my_tree
