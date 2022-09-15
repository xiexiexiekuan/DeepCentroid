import numpy
from sklearn import metrics

threshold = 0  # 分类阈值


# 划分数据为指定数量的训练集和验证集
def data_divide(n, bootstrap_size, partition_list, data, label, sign=True):
    times = data.shape[0]  # 抽样次数
    probability_ = probability(label)  # 获取抽样概率
    data_ = []  # 每个基因集合对应的训练集，训练标签，验证集，验证标签
    for j in range(0, n):
        tempt_data = []
        # bootstrap抽样，训练集约为63%
        index = numpy.random.choice(range(0, times), size=int(times * bootstrap_size), replace=True, p=probability_)
        index = numpy.unique(index)  # 去除重复抽样的标签
        difference = numpy.array(list(set(numpy.arange(0, times)) - set(index)))

        # sign=True 表示 partition_list 有效
        if sign:
            par = data[:, partition_list[j]]  # 根据基因集合提取出数据
        else:
            par = data
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
        array_ = gene_set_centroid(d[0], d[1], d[0].shape[1])  # 第三个参数是特征数
        array_centroid.append(array_)
    return numpy.array(array_centroid, dtype=object)


# 对每个基因计算质心的平均向量和权向量
def gene_set_centroid(gene_data, sample_category, gene_num):
    """
    质心分类器，对每个基因计算质心的平均向量和权向量
    :param gene_num: 基因数量
    :param gene_data: 所有基因表达值数据矩阵
    :param sample_category: 样本类别，将样本区分为正样本1和负样本0
    :return: 对M个基因都有2个数据，即M*2数据矩阵array_centroid
    """
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
            print('c_p_num除0异常')
        else:
            c_positive = p_sum[i] / c_p_num  # 计算基因表达平均值
        if c_n_num == 0:
            c_negative = 0
            print('c_n_num除0异常')
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
    # print(data.shape)
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
    score_l = []
    value = []
    weight = []
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
        # c = metrics.confusion_matrix(label, classifier)
        # fpr = (c[0][1] + c[1][0]) / (c[0][1] + c[1][0] + c[0][0] + c[1][1])
        score_l.append(mcc_score)

    i = 0
    for index, var in enumerate(score_l):
        if var > 0:
            value.append(index)
            weight.append(var)
        else:
            i = i + 1

    print('删除分类器数目 mcc < 0 : ', i)
    vector = numpy.array(centroid_vector_[value], dtype=object)
    plist = gene_partition_list[value]

    return vector, plist, weight


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
    return pro
