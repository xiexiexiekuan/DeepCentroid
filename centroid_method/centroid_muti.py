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
from sklearn.neural_network import MLPClassifier
from deepforest import CascadeForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost
from sklearn import svm

##########################################################################################
# This file corresponds to ablation experiment 2, multiple classifiers Integration testing, and the use method is to replace the centroid file of main.py file with centroid_ Muti is sufficient
# Note that there are many types of classifiers in this file, and pay attention to the installation environment. This article is Python 3.6

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
model = []


# initialization
def initialization(select, gene_set_num, gene_set_min, gene_set_max):
    partition.initialization(gene_set_num, gene_set_min, gene_set_max)
    centroid_compute.initialization(gene_set_num)
    global centroid_vector, gene_partition_list, extra_partition_list, level_data, layer, valid_layer, extra_vector, feature_select, is_feature_select, model
    centroid_vector = []
    gene_partition_list = []
    extra_partition_list = []
    level_data = []
    layer = 1
    valid_layer = 1
    extra_vector = []
    feature_select = []
    is_feature_select = select
    model = []


def model_score(y_ture, distance, sign=False):

    if sign:
        y_pred_p = numpy.int64(distance >= 0)
        y_pred_p = numpy.average(y_pred_p, axis=0)
        y_pred = numpy.int64(y_pred_p >= 0.5)
    else:
        y_pred_p = numpy.int64(distance >= 0.5)
        y_pred_p = numpy.average(y_pred_p, axis=0)
        y_pred = numpy.int64(y_pred_p >= 0.5)

    accuracy = metrics.accuracy_score(y_ture, y_pred)
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


def train(train_data_, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation):
    global layer, valid_layer, centroid_vector, gene_partition_list, extra_partition_list, level_data, probability, extra_vector, feature_select, model

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
            statistics_data = centroid_compute.gene_set_statistics(extra_list, train_data_)
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
            partition_list = partition.get_known_set(partition_list)
        partition_num = partition_list.shape[0]

        # Calculate the centroid vectors of all genes and store the output matrix of N * M * 2, that is, calculate the M * 2 variables related to the centroid for each gene set
        if layer == 1:
            data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, express_data, train_label, probability)
            gene_partition_list.append(partition_list)
            vector_ = centroid_compute.centroid_vector(partition_num, data_cut_set)

            # Remove bad classifiers based on the verification set verification results, only in the first layer
            if layer == 1 and cut_centroid:
                centroid_vector_, gene_partition_list_ = centroid_compute.verify_centroid_distance(partition_list, vector_, data_cut_set, False)  # 不进行基因注释
                centroid_vector = centroid_vector_  # Only one layer of centroid classifier results are retained here, so assignment is sufficient
                gene_partition_list = gene_partition_list_
            else:
                centroid_vector = vector_
                gene_partition_list = partition_list

            # Calculate the distance between the training set and the centroid for the next layer of training
            distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, express_data)
            # Preserve centroid distance for lower level training
            level_data = numpy.array(distance.T).astype(numpy.float32)
            result, y_pred = model_score(train_label, distance, True)
            print("Training set centroid prediction results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(result[0], result[4], result[5]), end="  ")
            print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))

        else:
            muti_level_data = []
            muti_model = []
            # Integrate multiple classifiers
            m = RandomForestClassifier()
            m.fit(express_data, train_label)
            muti_model.append(m)
            p = m.predict_proba(express_data)
            p = numpy.array(p[:, 1]).T
            muti_level_data.append(p)

            m = svm.SVC(kernel='rbf', C=10, probability=True)
            m.fit(express_data, train_label)
            muti_model.append(m)
            p = m.predict_proba(express_data)
            p = numpy.array(p[:, 1]).T
            muti_level_data.append(p)

            m = CascadeForestClassifier(verbose=0)
            m.fit(express_data, train_label)
            muti_model.append(m)
            p = m.predict_proba(express_data)
            p = numpy.array(p[:, 1]).T
            muti_level_data.append(p)

            dtrain = xgboost.DMatrix(express_data, label=train_label)
            param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic',
                     'eval_metric': ['logloss', 'auc', 'error']}  # Set parameters for XGB and pass them in dictionary format
            bst = xgboost.train(param, dtrain=dtrain)  # train
            muti_model.append(bst)
            p = bst.predict(xgboost.DMatrix(express_data))  # prediction
            muti_level_data.append(p)

            m = MLPClassifier(solver='lbfgs')
            m.fit(express_data, train_label)
            muti_model.append(m)
            p = m.predict_proba(express_data)
            p = numpy.array(p[:, 1]).T
            muti_level_data.append(p)

            model.append(muti_model)

            y_pred_p = numpy.average(muti_level_data, axis=0)
            y_pred = numpy.int64(y_pred_p >= 0.5)  # Prediction label

            accuracy = metrics.accuracy_score(train_label, y_pred)  # True and predicted values
            auc = metrics.roc_auc_score(train_label, y_pred_p)
            mcc_score_ = metrics.matthews_corrcoef(train_label, y_pred)
            print("Integrated training set prediction results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(accuracy, auc, mcc_score_), end="  ")
            level_data = numpy.array(muti_level_data).T

            print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))
            # Control the number of layers in the model and stop running when the MCC grows too little
            valid_layer = layer

            if mcc_score_ <= mcc:
                if mcc_score_ != 1.0:
                    valid_layer -= 1
                break
            else:
                mcc = mcc_score_
        layer += 1  # Add one layer to the number of layers


def predict(test_data, test_data_label):
    result_p = []
    result_set = []

    if is_feature_select:
        test_data = test_data[:, feature_select]
    c_distance = test_data
    c_distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, c_distance)
    result, y_pred_p = model_score(test_data_label, c_distance, True)
    print("{} layer test set prediction results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(1, result[0], result[4], result[5]))
    c_distance = numpy.array(c_distance.T).astype(numpy.float32)

    for v in range(1, valid_layer):
        result = []
        m1 = model[v-1]
        statistics_test = centroid_compute.gene_set_statistics(extra_partition_list[v - 1], test_data)
        c_distance = numpy.append(c_distance, statistics_test, axis=1)  # Splicing by column

        for i in range(0, 5):
            m = m1[i]
            if i == 3:
                dtest = xgboost.DMatrix(c_distance)
                y_pred_p = m.predict(dtest)
                result.append(y_pred_p)
            else:
                y_pred_p = m.predict_proba(c_distance)
                y_pred_p = y_pred_p[:, 1]
                y_pred_p = numpy.array(y_pred_p).T
                result.append(y_pred_p)

        c_distance = numpy.array(result).T
        result_p = numpy.mean(result, axis=0)
        result = numpy.int64(result_p >= 0.5)
        acc_score = metrics.accuracy_score(test_data_label, result)
        precision = metrics.precision_score(test_data_label, result)
        recall = metrics.recall_score(test_data_label, result)
        f1_score = metrics.f1_score(test_data_label, result)
        auc_score = metrics.roc_auc_score(test_data_label, result_p)
        mcc_score = metrics.matthews_corrcoef(test_data_label, result)
        print("{} layer test set prediction results ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(v + 1, acc_score, auc_score, mcc_score))

        result_set = numpy.array([acc_score, precision, recall, f1_score, auc_score, mcc_score])


    return result_set, test_data_label, y_pred_p
