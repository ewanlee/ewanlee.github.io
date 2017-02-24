from math import log
import operator
import numpy as np


def CalcShannon(data_set):
    """ Calculate the Shan0n Entropy.

    Arguments:
        data_set: The object dataset.

    Returns:
        shan0n_ent: The Shan0n entropy of the object data set.
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
    # print(label_counts)
    # Calculates the Shan0n entropy
    shan0n_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shan0n_ent -= prob * log(prob, 2)
    return shan0n_ent


def CalcGiniImpurity(data_set):
    """ Calculate the Gini impurity.

    Arguments:
        data_set: The object dataset.

    Returns:
        gini_impurity: The Gini impurity of the object data set.
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
    # Calculates the Gini impurity
    gini_impurity = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        gini_impurity += pow(prob, 2)
    gini_impurity = 1 - gini_impurity
    return gini_impurity


def CalcMisClassifyImpurity(data_set):
    """ Calculate the misclassification impurity.

    Arguments:
        data_set: The object dataset.

    Returns:
        mis_classify_impurity:
            The misclassification impurity of the object data set.
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
    # Calculates the misclassification impurity
    mis_classify_impurity = 0.0
    max_prob = max(label_counts.values()) / num_entries
    mis_classify_impurity = 1 - max_prob
    return mis_classify_impurity


def CreateDataSet(method='ID3'):
    """ A naive data generation method.

    Arguments:
        method: The algorithm class

    Returns:
        data_set: The data set excepts label info.
        labels: The data set only contains label info.
    """

    # Arguments check
    if method not in ('ID3', 'C4.5'):
        raise ValueError('invalid value: %s' % method)
    if method == 'ID3':
        data_set = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0]]
        labels = ['0 surfacing', 'flippers']
    else:
        data_set = [[1, 85, 85, 0, 0],
                    [1, 80, 90, 1, 0],
                    [2, 83, 78, 0, 1],
                    [3, 70, 96, 0, 1],
                    [3, 68, 80, 0, 1],
                    [3, 65, 70, 1, 0],
                    [2, 64, 65, 1, 1],
                    [1, 72, 95, 0, 0],
                    [1, 69, 70, 0, 1],
                    [3, 75, 80, 0, 1],
                    [1, 75, 70, 1, 1],
                    [2, 72, 90, 1, 1],
                    [2, 81, 75, 0, 1],
                    [3, 71, 80, 1, 0]]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
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


def ChooseBestFeatureToSplit(data_set, flag='ID3'):
    """ Choose the best feature to split.

    Arguments:
        data_set: Object data set.
        flag: Decide if use the infomation gain rate or not.

    Returns:
        best_feature: The index of the feature to split.
    """

    # Initiation
    # Because the range() method excepts the lastest number
    num_features = len(data_set[0]) - 1
    base_entropy = CalcShannon(data_set)
    method = 'ID3'
    best_feature = -1
    best_info_gain = 0.0
    best_info_gain_rate = 0.0

    for i in range(num_features):
        new_entropy = 0.0
        # Choose the i-th feature of all data
        feat_list = [example[i] for example in data_set]
        # Abandon the repeat feature value(s)
        unique_vals = set(feat_list)
        if len(unique_vals) > 3:
            method = 'C4.5'

        if method == 'ID3':
            # Calculates the Shannon entropy of the splited data set
            for value in unique_vals:
                sub_data_set = SplitDataSet(data_set, i, value)
                prob = len(sub_data_set) / len(data_set)
                new_entropy += prob * CalcShannon(sub_data_set)
        else:
            data_set = np.array(data_set)
            sorted_feat = np.argsort(feat_list)

            for index in range(len(sorted_feat) - 1):
                pre_sorted_feat, post_sorted_feat = np.split(
                    sorted_feat, [index + 1, ])
                pre_data_set = data_set[pre_sorted_feat]
                post_data_set = data_set[post_sorted_feat]
                pre_coff = len(pre_sorted_feat) / len(sorted_feat)
                post_coff = len(post_sorted_feat) / len(sorted_feat)
                # Calucate the split info
                iv = pre_coff * CalcShannon(pre_data_set) + \
                    post_coff * CalcShannon(post_data_set)
                if iv > new_entropy:
                    new_entropy = iv
        # base_entropy is equal or greatter than new_entropy
        info_gain = base_entropy - new_entropy
        if flag == 'C4.5':
            info_gain_rate = info_gain / new_entropy
            # print('index', i, 'info_gain_rate', info_gain_rate)
            if info_gain_rate > best_info_gain_rate:
                best_info_gain_rate = info_gain_rate
                best_feature = i
        if flag == 'ID3':
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

    return best_feature


def majority_cnt(class_list):
    """ Decided the final class.

    When the splited data is 0t belongs to the same class
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
        class_count.

        items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def CreateTree(data_set, feat_labels, method='ID3'):
    """ Create decision tree.

    Arguments:
        data_set: The object data set.
        labels: The feature labels in the data_set.
        method: The algorithm class.

    Returns:
        my_tree: A dict that represents the decision tree.
    """

    # Arguments check
    if method not in ('ID3', 'C4.5'):
        raise ValueError('invalid value: %s' % method)

    labels = feat_labels.copy()
    class_list = [example[-1] for example in data_set]
    # print(class_list)
    # If the classes are fully same
    print('class_list', class_list)
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # If all feature is handled
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    if method == 'ID3':
        # Get the best split-feature and the correspond label
        best_feat = ChooseBestFeatureToSplit(data_set)
        best_feat_label = labels[best_feat]
        # print(best_feat_label)
        # Build a recurrence dict
        my_tree = {best_feat_label: {}}
        # Next step start
        feat_values = [example[best_feat] for example in data_set]
        # Get the next step labels parameter
        del(labels[best_feat])
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            # Recurrence calls
            my_tree[best_feat_label][value] = CreateTree(
                SplitDataSet(data_set, best_feat, value), sub_labels)
        return my_tree
    else:
        flag = 'ID3'
        # Get the best split-feature and the correspond label
        best_feat = ChooseBestFeatureToSplit(data_set, 'C4.5')
        best_feat_label = labels[best_feat]
        print(best_feat_label)
        # Build a recurrence dict
        my_tree = {best_feat_label: {}}
        # Next step start
        feat_values = [example[best_feat] for example in data_set]
        del(labels[best_feat])
        unique_vals = set(feat_values)
        if len(unique_vals) > 3:
            flag = 'C4.5'

        if flag == 'ID3':
            for value in unique_vals:
                sub_labels = labels[:]
                # Recurrence calls
                my_tree[best_feat_label][value] = CreateTree(
                    SplitDataSet(data_set, best_feat, value),
                    sub_labels, 'C4.5')
            return my_tree
        else:
            data_set = np.array(data_set)
            best_iv = 0.0
            best_split_value = -1
            sorted_feat = np.argsort(feat_values)

            for i in range(len(sorted_feat) - 1):
                pre_sorted_feat, post_sorted_feat = np.split(
                    sorted_feat, [i + 1, ])
                pre_data_set = data_set[pre_sorted_feat]
                post_data_set = data_set[post_sorted_feat]
                pre_coff = len(pre_sorted_feat) / len(sorted_feat)
                post_coff = len(post_sorted_feat) / len(sorted_feat)
                # Calucate the split info
                iv = pre_coff * CalcShannon(pre_data_set) + \
                    post_coff * CalcShannon(post_data_set)
                if iv > best_iv:
                    best_iv = iv
                    best_split_value = feat_values[sorted_feat[i]]
            # print(best_feat, best_split_value)

            # print(best_split_value)

            left_data_set = data_set[
                data_set[:, best_feat] <= best_split_value]
            left_data_set = np.delete(left_data_set, best_feat, axis=1)
            # if len(left_data_set) == 1:
            #     return left_data_set[0][-1]
            right_data_set = data_set[
                data_set[:, best_feat] > best_split_value]
            right_data_set = np.delete(right_data_set, best_feat, axis=1)
            # if len(right_data_set) == 1:
            #     return right_data_set[0][-1]
            sub_labels = labels[:]
            my_tree[best_feat_label][
                '<=' + str(best_split_value)] = CreateTree(
                    left_data_set.tolist(), sub_labels, 'C4.5')
            my_tree[best_feat_label][
                '>' + str(best_split_value)] = CreateTree(
                    right_data_set.tolist(), sub_labels, 'C4.5')
            # print('continious tree', my_tree)
            return my_tree


def Classify(input_tree, feat_labels, test_vec):
    """ Classify that uses the given decision tree.

    Arguments:
        input_tree: The Given decision tree.
        feat_labels: The labels of correspond feature.
        test_vec: The test data.

    Returns:
        class_label: The class label that corresponds to the test data.
    """

    # Get the start feature label to split
    first_str = list(input_tree.keys())[0]
    # Get the sub-tree that corresponds to the start feature to split
    second_dict = input_tree[first_str]
    # Get the feature index that the label is the start feature label
    feat_index = feat_labels.index(first_str)

    # Start recurrence search
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                # Recurrence calls
                class_label = Classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]

    return class_label
