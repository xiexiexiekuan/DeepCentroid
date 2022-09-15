import time
import centroid_compute
import numpy
from sklearn import metrics
import partition
import randomforest
import simple_centroid
from centroid_compute import threshold

centroid_vector = []  # 第一层质心分类器生成的质心向量
gene_partition_list = []  # 质心分类器基因集合
extra_partition_list = []  # 额外数据基因集合
level_data = []  # 层之间传输数据
layer = 1  # 训练层数
valid_layer = 1  # 有效训练层数
weight = []  # 质心分类器权重
model = []  # 其他模型列表
forest_num = 1000  # 随机森林的数量


# 初始化
def initialization():
    global centroid_vector, gene_partition_list, extra_partition_list, level_data, layer, valid_layer, weight, model
    centroid_vector = []
    gene_partition_list = []
    extra_partition_list = []
    level_data = []
    layer = 1
    valid_layer = 1
    weight = []
    model = []


# 修改分类器权重
def modify_weights(data):
    global weight

    # softmax权重平滑
    def softmax(x):
        c = numpy.max(x)
        exp_a = numpy.exp(x - c)  # 溢出对策
        sum_exp_a = numpy.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    w = softmax(numpy.array(weight))
    weight_value = numpy.diag(w)  # 一维数组生成对角矩阵
    data_ = numpy.matmul(weight_value, data)  # 矩阵乘法，让data的每一行乘以一个系数，达到修改分类器权重
    return data_


# 计算集合的预测结果
def predict_result(data, y_ture):
    global level_data

    # 基因集合数x样本数
    centroid_distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, data)
    # centroid_distance = modify_weights(centroid_distance)  # 调整分类器权重

    # 保留质心距离用于下层训练
    level_data = numpy.array(centroid_distance.T).astype(numpy.float32)
    # 计算结果
    acc_score, auc_score, mcc_score = model_score(y_ture, centroid_distance)
    print("训练集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(acc_score, auc_score, mcc_score))

    return mcc_score


# 计算ACC, AUC, MCC
def model_score(y_ture, distance):

    y_pred_p = numpy.int64(distance >= threshold)  # 预测概率
    y_pred_p = numpy.average(y_pred_p, axis=0)
    y_pred = numpy.int64(y_pred_p >= 0.5)  # 预测标签

    acc_score = metrics.accuracy_score(y_ture, y_pred)  # 真实值，预测值
    auc_score = metrics.roc_auc_score(y_ture, y_pred_p)
    mcc_score = metrics.matthews_corrcoef(y_ture, y_pred)

    return acc_score, auc_score, mcc_score


# 根据verify验证集删除效果差的基因集合
def verify_test(data_cut_set, vector_, list_):
    global weight
    # centroid_vector的一维N的数量变少了，partition_list也是
    vector_, list_, weight_ = centroid_compute.verify_centroid_distance(list_, vector_, data_cut_set)
    weight = weight_
    return vector_, list_


# 随机森林分类器模型
def rf_class(data_cut_set):
    model_ = []
    for item in data_cut_set:
        m = randomforest.random_forest_model(item[0], item[1])
        model_.append(m)
    return model_


def train(train_data_, train_label, partition_list, max_train_layer=10, bootstrap_size=1):
    global layer, valid_layer, centroid_vector, gene_partition_list, extra_partition_list, level_data, forest_num

    layer_time = time.time()  # 第一层计时函数
    print('第 {} 层训练：'.format(layer))

    # 基因集合数目
    partition_num = partition_list.shape[0]

    # 获取每个基因集合对应的训练集，训练标签，验证集，验证标签
    data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, train_data_, train_label)

    # 计算所有基因的质心向量，保存输出的N*M*2的矩阵，即每个基因集合计算出关于质心的M*2个变量
    centroid_vector_ = centroid_compute.centroid_vector(partition_num, data_cut_set)

    # 根据验证集verify结果剔除不好的分类器，保留第一层的质心向量，保留第一层的基因集合
    # centroid_vector, gene_partition_list = verify_test(data_cut_set, centroid_vector_, partition_list)
    centroid_vector = centroid_vector_
    gene_partition_list = partition_list
    # 计算训练集与质心的距离用于下一层训练
    mcc = predict_result(train_data_, train_label)
    print("第 {} 层的计算时间：{:.2f} min.".format(layer, (time.time() - layer_time) / 60.0))

    # 级联训练
    layer += 1  # 模型层数，此时为2
    while True:
        if max_train_layer < layer:
            print('训练层数过高，训练结束！')
            break

        print('第 {} 层训练：'.format(layer))
        layer_time = time.time()  # 本层计算时间

        # 建立额外的拼接数据
        extra_list = partition.random_cut_data(train_data_.shape[1])
        extra_partition_list.append(extra_list)
        statistics_data = simple_centroid.sample_centroid_list(extra_list, train_data_, train_label)
        # 按列拼接上一层输出数据和新数据
        express_data = numpy.append(level_data, statistics_data, axis=1).astype(numpy.float32)
        print('输入数据：', express_data.shape)

        data_cut_set = centroid_compute.data_divide(forest_num, bootstrap_size, [], express_data, train_label, False)

        # 生成随机森林分类器模型
        model_ = rf_class(data_cut_set)
        model.append(model_)
        level_data, auc, mcc_score_ = randomforest.rf_predict_score("train", model_, express_data, train_label)

        # 控制模型最小层数，并当mcc增长过少时停止运行
        if mcc_score_ - mcc <= 0.01:
            break
        else:
            valid_layer = layer
            mcc = mcc_score_
        layer += 1  # 层数加一
        print("第 {} 层的计算时间：{:.2f} min.".format(layer, (time.time() - layer_time) / 60.0))


# 预测
def predict(test_data, test_data_label):
    c_distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, test_data)

    acc_score, auc_score, mcc_score = model_score(test_data_label, c_distance)
    print("测试集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(acc_score, auc_score, mcc_score))

    auc = 0
    mcc = 0
    # c_distance = modify_weights(c_distance)  # 调整分类器权重
    c_distance = numpy.array(c_distance.T).astype(numpy.float32)
    for i in range(0, valid_layer - 1):
        mod = model[i]
        statistics_test = simple_centroid.sample_centroid_list(extra_partition_list[i], test_data, test_data_label)
        c_distance = numpy.append(c_distance, statistics_test, axis=1)  # 按列拼接
        c_distance, auc, mcc = randomforest.rf_predict_score("test", mod, c_distance, test_data_label)

    return auc, mcc
