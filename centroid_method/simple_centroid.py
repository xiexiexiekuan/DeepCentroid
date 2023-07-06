import numpy
from sklearn import metrics


# Calculate centroid
def gene_centroid(gene_data, sample_category):
    gene_num = gene_data.shape[1]
    p_sample = numpy.where(sample_category == 1)[0]
    c_p_num = len(p_sample)
    p_data = gene_data[p_sample]  # Take a row
    p_sum = numpy.sum(p_data, axis=0)
    n_sample = numpy.where(sample_category == 0)[0]
    c_n_num = len(n_sample)
    n_data = gene_data[n_sample]  # Take a row
    n_sum = numpy.sum(n_data, axis=0)
    array_centroid = []
    for i in range(0, gene_num):
        if c_p_num == 0:
            c_positive = 0
            print('c_p_num 除0异常')
        else:
            c_positive = p_sum[i] / c_p_num  # Calculate average gene expression
        if c_n_num == 0:
            c_negative = 0
            print('c_n_num 除0异常')
        else:
            c_negative = n_sum[i] / c_n_num
        c_ = (c_positive + c_negative) / 2  # The average vector of the center of mass
        w_ = c_positive - c_negative  # weight vector
        # print("c = {}, w = {}".format(c, w))
        array_centroid.append([c_, w_])
    return numpy.array(array_centroid)


# Calculate the distance from the center of mass for each sample separately
def sample_centroid_distance(centroid_vector, data, b=False):
    sample_num = data.shape[0]

    inner_array = []  # Centroid distance vector of all samples
    for i in range(0, sample_num):  # Calculate each sample in sequence, totaling samples_ Num times
        inner_product = 0  # inner product
        for index in range(0, centroid_vector.shape[0]):
            d = centroid_vector[index]
            # if b:
            #     print(data.shape," ",i)
            inner_product += (data[i][index] - d[0]) * d[1]

        inner_array.append(inner_product)  # sample_num
    return numpy.array(inner_array)


# Simple centroid classifier
def sample_centroid(train_data, train_label, test_data, test_label):
    centroid_vector = gene_centroid(train_data, train_label)

    c_distance = sample_centroid_distance(centroid_vector, test_data)

    result = numpy.int64(c_distance >= 0)

    acc_score = metrics.accuracy_score(test_label, result)  # True value，Estimate value
    auc_score = metrics.roc_auc_score(test_label, result)
    mcc_score = metrics.matthews_corrcoef(test_label, result)
    print("Simple centroid  ACC = {:.3f}, AUC = {:.3f}, MCC = {:.3f}".format(acc_score, auc_score, mcc_score))
    return auc_score, mcc_score


# Calculate multiple simple centroid classifiers based on gene sets
def sample_centroid_list(extra_partition_list, train_data, train_label):
    result = []
    extra_vector = []

    for gene_set in extra_partition_list:
        # print(gene_set)
        data = train_data[:, gene_set]  # Extract data from this gene set into a new array
        centroid_vector = gene_centroid(data, train_label)

        c_distance = sample_centroid_distance(centroid_vector, data)

        extra_vector.append(centroid_vector)
        result.append(c_distance)
    result = numpy.array(result).T
    return result, extra_vector  # Sample x gene set


def sample_centroid_list_predict(extra_partition_list, train_data, centroid_vector_):
    result = []
    i = 0
    for gene_set in extra_partition_list:
        # print(gene_set)
        data = train_data[:, gene_set]  # Extract data from this gene set into a new array

        centroid_vector = centroid_vector_[i]
        i = i + 1
        c_distance = sample_centroid_distance(centroid_vector, data, True)
        result.append(c_distance)

    result = numpy.array(result).T
    return result  # Sample x gene set
