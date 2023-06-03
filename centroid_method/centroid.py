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

centroid_vector = []  # 第一层质心分类器生成的质心向量
gene_partition_list = []  # 质心分类器基因集合
extra_partition_list = []  # 额外数据基因集合
level_data = []  # 层之间传输数据
layer = 1  # 训练层数
valid_layer = 1  # 有效训练层数
extra_vector = []
probability = []
feature_select = []
is_feature_select = False  # 是否特征筛选


# 初始化
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


# 训练集计算集合的预测结果
def predict_result(data, y_ture, pro, l):
    global level_data, probability

    # 基因集合数x样本数
    distance = centroid_compute.sample_centroid_distance(gene_partition_list[l], centroid_vector[l], data)
    # print(distance)
    # 保留质心距离用于下层训练
    level_data = numpy.array(distance.T).astype(numpy.float32)

    # 计算结果
    result, y_pred = model_score(y_ture, distance, True)
    print("训练集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(result[0], result[4], result[5]), end="  ")

    # 计算新的抽样概率
    probability = centroid_compute.probability_weight(pro, y_ture, y_pred)

    return result[5]  # mcc


# 测试集计算ACC, AUC, MCC
def model_score(y_ture, distance, sign=False):


    y_pred_p = numpy.int64(distance >= 0)  # 预测概率
    # print(y_pred_p)
    y_pred_p = numpy.average(y_pred_p, axis=0)
    # print(y_pred_p)
    y_pred = numpy.int64(y_pred_p >= 0.5)  # 预测标签

    accuracy = metrics.accuracy_score(y_ture, y_pred)  # 真实值，预测值
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

# 训练
def train(train_data_, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation):
    global layer, valid_layer, centroid_vector, gene_partition_list, extra_partition_list, level_data, probability, extra_vector, feature_select

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
            statistics_data, vector_ = simple_centroid.sample_centroid_list(extra_list, train_data_, train_label)  # 使用简单质心作为特征计算方式
            # statistics_data = centroid_compute.gene_set_statistics(extra_list, train_data_)  # 使用平均值作为特征计算方式，该功能没有便捷接口，可在此修改，记得同时修改预测部分的
            extra_vector.append(vector_)
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
        # print(partition_num)

        data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, express_data, train_label, probability)

        # 计算所有基因的质心向量，保存输出的N*M*2的矩阵，即每个基因集合计算出关于质心的M*2个变量
        vector_ = centroid_compute.centroid_vector(partition_num, data_cut_set)

        # 根据验证集verify结果剔除不好的分类器，只在第一层
        if layer == 1 and cut_centroid:
            centroid_vector_, gene_partition_list_ = centroid_compute.verify_centroid_distance(partition_list, vector_, data_cut_set, function_annotation)
            centroid_vector.append(centroid_vector_)
            gene_partition_list.append(gene_partition_list_)
        else:
            centroid_vector.append(vector_)
            gene_partition_list.append(partition_list)

        # 计算训练集与质心的距离用于下一层训练
        mcc_score_ = predict_result(express_data, train_label, probability, layer-1)
        print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))

        # 控制模型层数，并当mcc增长过少时停止运行
        valid_layer = layer

        if mcc_score_ <= mcc:
            # 如果最后一层 mcc=1 就保留当层
            if mcc_score_ != 1.0:
                valid_layer -= 1
            break
        else:
            mcc = mcc_score_
        layer += 1  # 层数加一


# 预测
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
            c_distance = numpy.append(c_distance, statistics_test, axis=1)  # 按列拼接

        c_distance = centroid_compute.sample_centroid_distance(gene_partition_list[i], centroid_vector[i], c_distance)
        result, y_pred_p = model_score(test_data_label, c_distance)
        print("{} 层测试集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(i+1, result[0], result[4], result[5]))

        c_distance = numpy.array(c_distance.T).astype(numpy.float32)

    return result, test_data_label, y_pred_p
