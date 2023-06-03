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
from sklearn.feature_selection import SelectFromModel
from sklearn import svm

##########################################################################################
# 此文件对应消融实验1，随机森林集成测试，使用方式为将main.py文件的centroid文件替换为centroid_rf即可

centroid_vector = []  # 第一层质心分类器生成的质心向量--本文件未使用
gene_partition_list = []  # 质心分类器基因集合
extra_partition_list = []  # 额外数据基因集合
level_data = []  # 层之间传输数据
layer = 1  # 训练层数
valid_layer = 1  # 有效训练层数
extra_vector = []
probability = []
feature_select = []
is_feature_select = False  # 是否特征筛选
tree_model = []

# 初始化
# 初始化
def initialization(select, gene_set_num, gene_set_min, gene_set_max):
    partition.initialization(gene_set_num, gene_set_min, gene_set_max)
    centroid_compute.initialization(gene_set_num)
    global centroid_vector, gene_partition_list, extra_partition_list, level_data, layer, valid_layer, extra_vector, feature_select, is_feature_select, tree_model
    centroid_vector = []
    gene_partition_list = []
    extra_partition_list = []
    level_data = []
    layer = 1
    valid_layer = 1
    extra_vector = []
    feature_select = []
    is_feature_select = select
    tree_model = []


# 训练
def train(train_data_, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation):
    global layer, valid_layer, gene_partition_list, extra_partition_list, level_data, probability, extra_vector, feature_select, tree_model

    if is_feature_select:
        data = SelectFromModel(RandomForestClassifier()).fit(train_data_, train_label)
        feature_select = data.get_support()
        # print(train_data_.shape)
        train_data_ = train_data_[:, feature_select]
        print('feature_select: ', train_data_.shape)

    # 获取每个基因集合对应的训练集，训练标签，验证集，验证标签
    probability = centroid_compute.probability(train_label)
    mcc = 0
    express_data = train_data_

    # 级联训练
    while True:
        if max_train_layer < layer:
            print('训练层数过高，训练结束！')
            break

        print('第 {} 层训练：'.format(layer), end=" ")
        layer_time = time.time()  # 本层计算时间

        if layer >= 2:
            if not is_sliding_scan:
                extra_list = partition.random_cut_data(train_data_.shape[1])
            else:
                extra_list = partition.random_cut_data_order(train_data_.shape[1])
            extra_partition_list.append(extra_list)
            statistics_data = centroid_compute.gene_set_statistics(extra_list, train_data_)
            # 按列拼接上一层输出数据和新数据
            express_data = numpy.append(level_data, statistics_data, axis=1).astype(numpy.float32)

        print("输入数据：", express_data.shape)

        # 基因集合数目
        if not is_sliding_scan:
            partition_list = partition.random_cut_data(express_data.shape[1])  # 划分随机基因集合
        else:
            partition_list = partition.random_cut_data_order(express_data.shape[1])  # 划分随机基因集合
        # 如果添加已知基因集合，那么在第一层时添加
        if layer == 1 and known_feature_set:
            partition_list = partition.get_known_set(partition_list)  # 获取已知基因集合
        partition_num = partition_list.shape[0]

        data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, express_data, train_label, probability)
        gene_partition_list.append(partition_list)

        # 未提供剪枝与基因注释功能
        model = []
        tree_level_data = []
        for index, item in enumerate(data_cut_set):
            m = RandomForestClassifier()
            m.fit(item[0], item[1])
            p = m.predict_proba(express_data[:, partition_list[index]])
            p = numpy.array(p[:, 1]).T
            tree_level_data.append(p)
            model.append(m)
        tree_model.append(model)

        y_pred_p = numpy.average(tree_level_data, axis=0)
        y_pred = numpy.int64(y_pred_p >= 0.5)  # 预测标签

        accuracy = metrics.accuracy_score(train_label, y_pred)  # 真实值，预测值
        auc = metrics.roc_auc_score(train_label, y_pred_p)
        mcc_score_ = metrics.matthews_corrcoef(train_label, y_pred)
        print("训练集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(accuracy, auc, mcc_score_), end="  ")
        level_data = numpy.array(tree_level_data).T

        print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))
        # 控制模型层数，并当mcc增长过少时停止运行
        valid_layer = layer

        if mcc_score_ <= mcc or layer == 1:
            # valid_layer -= 1
            break
        else:
            mcc = mcc_score_
        layer += 1  # 层数加一


# 预测
def predict(test_data, test_data_label):
    result_p = []
    result_set = []

    if is_feature_select:
        test_data = test_data[:, feature_select]
    c_distance = test_data

    for i in range(0, valid_layer):
        result = []
        if i > 0:
            statistics_test = centroid_compute.gene_set_statistics(extra_partition_list[i - 1], test_data)
            c_distance = numpy.append(c_distance, statistics_test, axis=1)  # 按列拼接

        partition_list = gene_partition_list[i]
        for index, model in enumerate(tree_model[i]):
            result_ = model.predict_proba(c_distance[:, partition_list[index]])
            result_ = numpy.array(result_[:, 1]).T
            result.append(result_)

        c_distance = numpy.array(result).T
        result_p = numpy.mean(result, axis=0)
        result = numpy.int64(result_p >= 0.5)
        acc_score = metrics.accuracy_score(test_data_label, result)  # 真实值，预测值
        precision = metrics.precision_score(test_data_label, result)
        recall = metrics.recall_score(test_data_label, result)
        f1_score = metrics.f1_score(test_data_label, result)
        auc_score = metrics.roc_auc_score(test_data_label, result_p)
        mcc_score = metrics.matthews_corrcoef(test_data_label, result)
        print("{} 层测试集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(i + 1, acc_score, auc_score, mcc_score))

        result_set = numpy.array([acc_score, precision, recall, f1_score, auc_score, mcc_score])

    return result_set, test_data_label, result_p
