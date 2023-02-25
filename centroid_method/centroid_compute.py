from math import exp, pow, e
import numpy
import heapq
import pandas
from sklearn import metrics
import gene_annotation

threshold = 0  # 分类阈值
times_ = 0
best_gene_set = {}



# 划分数据为指定数量的训练集和验证集
def data_divide(n, bootstrap_size, partition_list, data, label, pro):
    times = data.shape[0]  # 抽样次数
    data_ = []  # 每个基因集合对应的训练集，训练标签，验证集，验证标签
    for j in range(0, n):
        tempt_data = []
        # bootstrap抽样，训练集约为63%   此处为不放回
        index = numpy.random.choice(range(0, times), size=int(times * bootstrap_size), replace=False, p=pro)
        index = numpy.unique(index)  # 去除重复抽样的标签
        difference = numpy.array(list(set(numpy.arange(0, times)) - set(index)))

        p_l = partition_list[j]
        p_l = numpy.array(p_l, dtype=int)
        # print(p_l)
        par = data[:, p_l]  # 根据基因集合提取出数据

        tempt_data.append(par[index])
        tempt_data.append(label[index])
        tempt_data.append(par[difference])
        tempt_data.append(label[difference])
        data_.append(tempt_data)
    return data_


# 计算所有基因的质心向量，保存输出的N*M*2的矩阵，即每个基因集合计算出关于质心的M*2个变量
def centroid_vector(partition_num, data_cut_set):
    array_centroid = []
    # 遍历N个基因集合，每个集合里面有M个基因，集合大小是可变的
    for i in range(0, partition_num):
        # print(var)
        d = data_cut_set[i]
        array_ = gene_set_centroid(d[0], d[1])  # 第三个参数是特征数
        array_centroid.append(array_)
    return numpy.array(array_centroid, dtype=object)


# 对每个基因计算质心的平均向量和权向量
def gene_set_centroid(gene_data, sample_category):
    """
    质心分类器，对每个基因计算质心的平均向量和权向量
    :param gene_data: 所有基因表达值数据矩阵
    :param sample_category: 样本类别，将样本区分为正样本1和负样本0
    :return: 对M个基因都有2个数据，即M*2数据矩阵array_centroid
    """
    gene_num = gene_data.shape[1]
    p_sample = numpy.where(sample_category == 1)[0]
    c_p_num = len(p_sample)
    p_data = gene_data[p_sample]  # 取行
    p_sum = numpy.sum(p_data, axis=0)
    n_sample = numpy.where(sample_category == 0)[0]
    c_n_num = len(n_sample)
    n_data = gene_data[n_sample]  # 取行
    n_sum = numpy.sum(n_data, axis=0)
    array_centroid = []
    for i in range(0, gene_num):
        if c_p_num == 0:
            c_positive = 0
            print('c_p_num 除0异常')
        else:
            c_positive = p_sum[i] / c_p_num  # 计算基因表达平均值
        if c_n_num == 0:
            c_negative = 0
            print('c_n_num 除0异常')
        else:
            c_negative = n_sum[i] / c_n_num
        c_ = (c_positive + c_negative) / 2  # 质心的平均向量
        w_ = c_positive - c_negative  # 权向量
        # print("c = {}, w = {}".format(c, w))
        array_centroid.append([c_, w_])
    return array_centroid


# 分别对每个样本计算其与质心的距离
def sample_centroid_distance(gene_partition_list, centroid_vector_, data):
    sample_num = data.shape[0]
    print(data.shape)
    inner_array = []  # 所有样本的质心距离向量
    for i in range(0, sample_num):  # 依次计算每个样本，共sample_num次
        inner_p_array = []  # 一个样本的质心距离向量
        for index, gene_set in enumerate(gene_partition_list):  # 按行遍历基因集合，共N次
            inner_product = 0  # 内积
            for index_k, k in enumerate(gene_set):  # 遍历每个基因集合的下角标，共M次
                # print(gene_set)
                d = centroid_vector_[index][index_k]
                inner_product += (data[i][k] - d[0]) * d[1]
            inner_p_array.append(inner_product)  # N*1的矩阵
        # print(inner_p_array)
        inner_array.append(inner_p_array)  # sample_num*N*1的矩阵
    distance = numpy.transpose(inner_array)  # N*sample_num*1的矩阵

    return distance


# 分别对每个样本计算其与质心的距离--剪枝
def verify_centroid_distance(gene_partition_list, centroid_vector_, data):

    list_num = gene_partition_list.shape[0]
    value = []
    i = 0
    i_ = 0
    score_l = []
    err_set = []
    best_set = []
    for k in range(0, list_num):
        inner_array = []  # 一个分类器的所有验证样本的质心距离向量   1*样本的矩阵
        data_ = data[k]
        verify = data_[2]  # 样本 x 特征
        label = data_[3]
        c_vector = centroid_vector_[k]  # 特征 x 2
        sample_num = len(label)
        for j in range(0, sample_num):
            inner_product = inner(verify[j], c_vector)
            inner_array.append(inner_product)

        inner_array = numpy.array(inner_array)
        classifier = numpy.int64(inner_array >= threshold)
        mcc_score = metrics.matthews_corrcoef(label, classifier)
        acc_score = metrics.accuracy_score(label, classifier)  # 真实值，预测值
        score_l.append(mcc_score)

        best_set.append(mcc_score)
        # print(mcc_score)
        if mcc_score >= 0:
            value.append(k)
            i_ += 1
            # err = 0.5*exp(acc_score/(1-acc_score))
            # print(1-acc_score)
            # err_set.append(err)
        else:
            i = i + 1

    # num = numpy.sum(err_set)
    # for i in range(len(err_set)):
    #     err_set[i] = err_set[i] / num
    # score_o = sorted(score_l)
    # print(score_o[int(len(score_o) * 0.5)], score_o[int(len(score_o))-1])
    # limit_new = score_o[int(len(score_o) * 0.5)]
    # i = 0
    #
    # for index, var in enumerate(score_l):
    #     if var >= limit_new:
    #         value.append(index)
    #         i_ += 1
    #     else:
    #         i += 1

    # global times_, best_gene_set
    # arr_max = heapq.nlargest(500, best_set)  # 获取前五大的值并排序
    # index_max = map(best_set.index, arr_max)  # 获取前五大的值下标
    #
    # for h in list(index_max):
    #     data = numpy.array(gene_partition_list[h])
    #     for d in data:
    #         if d in best_gene_set:
    #             best_gene_set[d] = best_gene_set[d] + 1
    #         else:
    #             best_gene_set[d] = 1
    # # data = pandas.DataFrame(data)
    # # data.to_csv('best_set'+str(times_)+'.csv', encoding='utf-8')
    # times_ += 1
    # if times_ == 5:
    #     times_ = 0
    #     gene_annotation.best_gene_symbol_prognosis(best_gene_set)
    #     # gene_annotation.best_gene_symbol_drug(best_gene_set)
    #     # gene_annotation.best_gene_symbol_liver(best_gene_set)

    print('删除分类器数目 mcc < 0 : ', i, i_)
    vector = numpy.array(centroid_vector_[value], dtype=object)
    plist = gene_partition_list[value]

    # num = numpy.sum(err_set)
    # for i in range(len(err_set)):
    #     err_set[i] = err_set[i] / num

    return vector, plist, numpy.diag(err_set)


# 一维数组和 特征x2的数组内积
def inner(v_data, c_vector):
    num = len(v_data)
    s = 0
    for p in range(0, num):
        d = c_vector[p]
        s += (v_data[p] - d[0]) * d[1]

    return s


# 根据正负样本比例计算抽样概率
def probability(label):
    l = len(label)
    p_num = sum(label)
    pro = numpy.zeros(l, dtype=float)
    # print(pro)
    p_p = 1 / (2 * p_num)
    n_p = 1 / (2 * (l - p_num))
    for i in range(0, l):
        if label[i] == 0:
            pro[i] = n_p
        else:
            pro[i] = p_p
    # print(pro)

    # p = 1.0/l
    # pro = []
    # for i in range(0, l):
    #     pro.append(p)
    return pro


# 根据正负样本比例计算抽样概率
def probability_weight(pro, y_ture, y_pred):
    num = 0
    for i in range(len(y_ture)):
        if y_ture[i] != y_pred[i]:
            num += 1
    err = num / len(y_ture)  # 平均错误率
    if err == 0:
        return pro

    for i in range(len(pro)):
        if y_ture[i] != y_pred[i]:
            pro[i] = 0.5 * pro[i] / err
        else:
            pro[i] = 0.5 * pro[i] / (1 - err)

    num = numpy.sum(pro)
    for i in range(len(pro)):
        pro[i] = pro[i] / num

    return pro


# 根据基因集合计算平均值，中位数
def gene_set_statistics(gene_partition_list, express_data):
    statistics = []
    for gene_set in gene_partition_list:  # 遍历N个基因集合，每个集合里面有M个基因
        # print(gene_set)
        gene_list = express_data[:, gene_set]  # 根据该基因集合将数据提取出来成为新的数组
        # print(gene_list.shape)
        # mean = numpy.mean(gene_list, axis=0)  # 压缩行，按列计算每个样本的平均数
        mean = numpy.mean(gene_list, axis=1)  # 压缩列，按行计算每个样本的平均数
        # median = numpy.median(gene_list, axis=0)  # 压缩行，按列计算每个样本的中位数
        # print('mean:', mean.shape)
        statistics.append(numpy.array(mean))
        # statistics.append(numpy.array(median))
    statistics = numpy.array(statistics)
    # print(statistics.shape)
    return statistics.T
