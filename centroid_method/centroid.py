import time
import pandas
import centroid_compute
import numpy
from sklearn import metrics
import partition
import simple_centroid
from centroid_compute import threshold
from sklearn.preprocessing import scale, normalize, minmax_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

centroid_vector = []  # The centroid vector generated by the first layer centroid classifier
gene_partition_list = []  # Centroid classifier gene set
extra_partition_list = []  # Additional data gene set
level_data = []  # Transfer data between layers
layer = 1  # Number of training layers
valid_layer = 1  # Effective training layers
extra_vector = []
probability = []
feature_select = []
is_feature_select = False  # Feature filtering or not


# initialization
def initialization(select, gene_set_num, gene_set_min, gene_set_max):
    partition.initialization(gene_set_num, gene_set_min, gene_set_max)
    centroid_compute.initialization(gene_set_num)
    global centroid_vector, gene_partition_list, extra_partition_list, level_data, layer, valid_layer, extra_vector, feature_select, is_feature_select
    centroid_vector = []
    gene_partition_list = []
    extra_partition_list = []
    level_data = []
    layer = 1
    valid_layer = 1
    extra_vector = []
    feature_select = []
    is_feature_select = select


# Prediction results of training set calculation set
def predict_result(data, y_ture, pro, l):
    global level_data, probability

    # Number of gene sets x number of samples
    distance = centroid_compute.sample_centroid_distance(gene_partition_list[l], centroid_vector[l], data)
    # print(distance)
    # Preserve centroid distance for lower level training
    level_data = numpy.array(distance.T).astype(numpy.float32)

    # Calculation results
    result, y_pred = model_score(y_ture, distance, True)
    print("Training Set Prediction Results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(result[0], result[4], result[5]), end="  ")

    # Calculate the new sampling probability
    probability = centroid_compute.probability_weight(pro, y_ture, y_pred)

    return result[5]  # mcc


# Test Set CalculationACC, AUC, MCC
def model_score(y_ture, distance, sign=False):


    y_pred_p = numpy.int64(distance >= 0)  # Predicted Probability
    # print(y_pred_p)
    y_pred_p = numpy.average(y_pred_p, axis=0)
    # print(y_pred_p)
    y_pred = numpy.int64(y_pred_p >= 0.5)  # Prediction label

    accuracy = metrics.accuracy_score(y_ture, y_pred)  # True and predicted values
    precision = metrics.precision_score(y_ture, y_pred)
    recall = metrics.recall_score(y_ture, y_pred)
    f1_score = metrics.f1_score(y_ture, y_pred)
    auc = metrics.roc_auc_score(y_ture, y_pred_p)
    mcc = metrics.matthews_corrcoef(y_ture, y_pred)

    result_set = numpy.array([accuracy, precision, recall, f1_score, auc, mcc])

    if sign:
        return result_set, y_pred
    else:
        return result_set, y_pred_p

# train
def train(train_data_, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation):
    global layer, valid_layer, centroid_vector, gene_partition_list, extra_partition_list, level_data, probability, extra_vector, feature_select

    if is_feature_select:
        data = SelectFromModel(RandomForestClassifier()).fit(train_data_, train_label)
        feature_select = data.get_support()
        # print(train_data_.shape)
        train_data_ = train_data_[:, feature_select]
        print('feature_select: ', train_data_.shape)

    # Obtain the training set, training label, validation set, and validation label corresponding to each gene set
    probability = centroid_compute.probability(train_label)
    mcc = 0
    express_data = train_data_

    # Cascading training
    while True:
        if max_train_layer < layer:
            print('The training level is too high, the training is over!')
            break

        print('{} layer training:'.format(layer), end=" ")
        layer_time = time.time()  # Calculation time of this layer

        if layer >= 2:
            if not is_sliding_scan:
                extra_list = partition.random_cut_data(train_data_.shape[1])
            else:
                extra_list = partition.random_cut_data_order(train_data_.shape[1])
            extra_partition_list.append(extra_list)
            statistics_data, vector_ = simple_centroid.sample_centroid_list(extra_list, train_data_, train_label)  # Using simple centroids as feature calculation methods
            # statistics_data = centroid_compute.gene_set_statistics(extra_list, train_data_)  # Using the average value as the feature calculation method, this function does not have a convenient interface and can be modified here. Remember to modify the prediction part at the same time
            extra_vector.append(vector_)
            # Splice the previous layer of output data and new data by column
            express_data = numpy.append(level_data, statistics_data, axis=1).astype(numpy.float32)

        print("data in：", express_data.shape)

        # Number of gene sets
        if not is_sliding_scan:
            partition_list = partition.random_cut_data(express_data.shape[1])  # Divide a random gene set
        else:
            partition_list = partition.random_cut_data_order(express_data.shape[1])  # Divide a random gene set
        # If adding a set of known genes, add them in the first layer
        if layer == 1 and known_feature_set:
            partition_list = partition.get_known_set(partition_list)  # Obtain a set of known genes

        partition_num = partition_list.shape[0]
        # print(partition_num)

        data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, express_data, train_label, probability)

        # Calculate the centroid vectors of all genes and store the output matrix of N * M * 2, that is, calculate the M * 2 variables related to the centroid for each gene set
        vector_ = centroid_compute.centroid_vector(partition_num, data_cut_set)

        # Remove bad classifiers based on the verification set verification results, only in the first layer
        if layer == 1 and cut_centroid:
            centroid_vector_, gene_partition_list_ = centroid_compute.verify_centroid_distance(partition_list, vector_, data_cut_set, function_annotation)
            centroid_vector.append(centroid_vector_)
            gene_partition_list.append(gene_partition_list_)
        else:
            centroid_vector.append(vector_)
            gene_partition_list.append(partition_list)

        # Calculate the distance between the training set and the centroid for the next layer of training
        mcc_score_ = predict_result(express_data, train_label, probability, layer-1)
        print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))

        # Control the number of layers in the model and stop running when the MCC grows too little
        valid_layer = layer

        if mcc_score_ <= mcc:
            # If the last layer mcc=1, keep the current layer
            if mcc_score_ != 1.0:
                valid_layer -= 1
            break
        else:
            mcc = mcc_score_
        layer += 1  # Add one layer to the number of layers


# predict
def predict(test_data, test_data_label):
    global extra_vector, feature_select

    if is_feature_select:
        test_data = test_data[:, feature_select]

    result = []
    y_pred_p = []
    c_distance = test_data
    for i in range(0, valid_layer):
        # print(gene_partition_list[i].shape)
        if i > 0:
            statistics_test = simple_centroid.sample_centroid_list_predict(extra_partition_list[i-1], test_data, extra_vector[i-1])
            # statistics_test = centroid_compute.gene_set_statistics(extra_partition_list[i - 1], test_data)
            c_distance = numpy.append(c_distance, statistics_test, axis=1)  # Splicing by column

        c_distance = centroid_compute.sample_centroid_distance(gene_partition_list[i], centroid_vector[i], c_distance)
        result, y_pred_p = model_score(test_data_label, c_distance)
        print("{} layer test set prediction results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(i+1, result[0], result[4], result[5]))

        c_distance = numpy.array(c_distance.T).astype(numpy.float32)

    return result, test_data_label, y_pred_p
