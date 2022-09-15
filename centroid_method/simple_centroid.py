import numpy
from sklearn import metrics


# 计算质心
def gene_centroid(gene_data, sample_category):
    sample_num = gene_data.shape[0]
    array_centroid = []
    c_positive, c_negative, c_p_num, c_n_num = 0, 0, 0, 0  # 正样本数据和，负样本数据和，正样本数，负样本数
    for i in range(0, gene_data.shape[1]):
        for j in range(0, sample_num):  # 对1个基因，有sample_num个样本的数据要计算
            if sample_category[j] == 1:
                c_p_num += 1  # 统计正样本数量
                c_positive += gene_data[j][i]  # 计算正样本表达值的和
            else:  # sample_category[j][0] == 0:
                c_n_num += 1
                c_negative += gene_data[j][i]

        c_positive = c_positive / c_p_num  # 计算基因表达平均值
        c_negative = c_negative / c_n_num
        # print("positive = {}, p_num = {}, negative = {}, n_num = {}".format(c_positive, c_negative, c_p_num, c_n_num))
        c_ = (c_positive + c_negative) / 2  # 质心的平均向量
        w_ = c_positive - c_negative  # 权向量
        # print("c = {}, w = {}".format(c, w))
        array_centroid.append([c_, w_])
    return numpy.array(array_centroid)


# 分别对每个样本计算其与质心的距离
def sample_centroid_distance(centroid_vector, data):
    sample_num = data.shape[0]
    # print(data.shape)
    inner_array = []  # 所有样本的质心距离向量
    for i in range(0, sample_num):  # 依次计算每个样本，共sample_num次
        inner_product = 0  # 内积
        for index in range(0, centroid_vector.shape[0]):
            d = centroid_vector[index]
            inner_product += (data[i][index] - d[0]) * d[1]

        inner_array.append(inner_product)  # sample_num的矩阵
    return numpy.array(inner_array)


# 简单质心分类器
def sample_centroid(train_data, train_label, test_data, test_label):
    centroid_vector = gene_centroid(train_data, train_label)

    c_distance = sample_centroid_distance(centroid_vector, test_data)

    result = numpy.int64(c_distance >= 0)

    acc_score = metrics.accuracy_score(test_label, result)  # 真实值，预测值
    auc_score = metrics.roc_auc_score(test_label, result)
    mcc_score = metrics.matthews_corrcoef(test_label, result)
    print("简单质心  ACC = {:.3f}, AUC = {:.3f}, MCC = {:.3f}".format(acc_score, auc_score, mcc_score))
    return auc_score, mcc_score


# 根据基因集合计算出多个简单质心分类器
def sample_centroid_list(extra_partition_list, train_data, train_label):
    result = []
    for gene_set in extra_partition_list:
        # print(gene_set)
        data = train_data[:, gene_set]  # 根据该基因集合将数据提取出来成为新的数组
        centroid_vector = gene_centroid(data, train_label)
        c_distance = sample_centroid_distance(centroid_vector, data)
        result.append(c_distance)
    result = numpy.array(result).T
    return result  # 样本x基因集合
