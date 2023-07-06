from math import exp, pow, e
import numpy
import heapq
import pandas
from sklearn import metrics
import gene_annotation

threshold = 0  # Classification threshold
times_ = 0
best_gene_set = {}
gene_set_num = 0

def initialization(set_num):
    global gene_set_num
    gene_set_num = set_num



# Divide the data into a specified number of training and validation sets
def data_divide(n, bootstrap_size, partition_list, data, label, pro):
    times = data.shape[0]  # Sampling frequency
    data_ = []  # Training set, training label, validation set, validation label corresponding to each gene set
    for j in range(0, n):
        tempt_data = []
        # bootstrapSampling, training set approximately 63%
        index = numpy.random.choice(range(0, times), size=int(times * bootstrap_size), replace=False, p=pro)
        index = numpy.unique(index)  # Removing labels for duplicate sampling
        difference = numpy.array(list(set(numpy.arange(0, times)) - set(index)))


        par = data[:, partition_list[j]]  # Extracting data from gene sets

        tempt_data.append(par[index])
        tempt_data.append(label[index])
        tempt_data.append(par[difference])
        tempt_data.append(label[difference])
        data_.append(tempt_data)
    return data_


# Calculate the centroid vectors of all genes and store the output matrix of N * M * 2, that is, calculate the M * 2 variables related to the centroid for each gene set
def centroid_vector(partition_num, data_cut_set):
    array_centroid = []
    # Traverse N gene sets, each containing M genes, with variable set sizes
    for i in range(0, partition_num):
        # print(var)
        d = data_cut_set[i]
        array_ = gene_set_centroid(d[0], d[1])  # The third parameter is the number of features
        array_centroid.append(array_)
    return numpy.array(array_centroid, dtype=object)


# Calculate the average vector and weight vector of the centroid for each gene
def gene_set_centroid(gene_data, sample_category):
    """
    Centroid classifier, which calculates the average vector and weight vector of the centroid for each gene
    :param gene_data: All gene expression value data matrix
    :param sample_category: Sample category, distinguishing samples into positive sample 1 and negative sample 0
    :return: There are 2 data for all M genes, namely the M * 2 data matrix array_ Centroid
    """
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
            print('c_p_num Exception except for 0')
        else:
            c_positive = p_sum[i] / c_p_num  # Calculate average gene expression
        if c_n_num == 0:
            c_negative = 0
            print('c_n_num Exception except for 0')
        else:
            c_negative = n_sum[i] / c_n_num
        c_ = (c_positive + c_negative) / 2  # The average vector of the center of mass
        w_ = c_positive - c_negative  # weight vector
        # print("c = {}, w = {}".format(c, w))
        array_centroid.append([c_, w_])
    return numpy.array(array_centroid)


def sample_centroid_distance_alone(centroid_vector, data):
    sample_num = data.shape[0]

    inner_array = []  # Centroid distance vector of all samples
    for i in range(0, sample_num):  # Calculate each sample in sequence, totaling samples_ Num times
        inner_product = 0  # inner product
        for index in range(0, centroid_vector.shape[0]):
            d = centroid_vector[index]
            inner_product += (data[i][index] - d[0]) * d[1]

        inner_array.append(inner_product)  # Sample_ Matrix of num
    return numpy.array(inner_array)

# Calculate the distance from the center of mass for each sample separately
def sample_centroid_distance(gene_partition_list, centroid_vector_, data):
    inner_array = []  # Centroid distance vector of all samples
    for index, gene_set in enumerate(gene_partition_list):  # Traverse the gene set by row, N times in total
        data_ = data[:, gene_set]
        # print(centroid_vector_[index].shape)
        # print(data_.shape)
        inner_p_array = sample_centroid_distance_alone(centroid_vector_[index], data_)
        inner_array.append(inner_p_array)  # sample_num*N*1 matrix

    distance = numpy.array(inner_array)  # N*sample_num*1 matrix

    return distance




# Calculate the distance from the center of mass for each sample separately - pruning
def verify_centroid_distance(gene_partition_list, centroid_vector_, data, function_annotation):

    list_num = gene_partition_list.shape[0]
    value = []
    i = 0
    i_ = 0
    score_l = []
    best_set = []
    for k in range(0, list_num):
        inner_array = []  # The centroid distance vector 1 * matrix of all validation samples for a classifier
        data_ = data[k]
        verify = data_[2]  # Sample * Features
        label = data_[3]
        c_vector = centroid_vector_[k]  # Features * 2
        sample_num = len(label)
        for j in range(0, sample_num):
            inner_product = inner(verify[j], c_vector)
            inner_array.append(inner_product)

        inner_array = numpy.array(inner_array)
        classifier = numpy.int64(inner_array >= threshold)
        mcc_score = metrics.matthews_corrcoef(label, classifier)
        score_l.append(mcc_score)

        best_set.append(mcc_score)
        # print(mcc_score)
        if mcc_score >= 0:
            value.append(k)
            i_ += 1
        else:
            i = i + 1

    if function_annotation:  # If functional comments
        global times_, best_gene_set, gene_set_num
        arr_max = heapq.nlargest(gene_set_num, best_set)  # Obtain the top number of values and sort them
        index_max = map(best_set.index, arr_max)  # Get the index of the top number of values

        for h in list(index_max):
            data = numpy.array(gene_partition_list[h])
            for d in data:
                if d in best_gene_set:
                    best_gene_set[d] = best_gene_set[d] + 1
                else:
                    best_gene_set[d] = 1
        # data = pandas.DataFrame(data)
        # data.to_csv('best_set'+str(times_)+'.csv', encoding='utf-8')
        times_ += 1
        if times_ == 5:  # Output one file for a 50% cross validation
            times_ = 0
###############################################################################
# Due to different data sources and processing methods, the three functions correspond to three types of data and require manual modification by the user
            gene_annotation.best_gene_symbol_prognosis(best_gene_set)  # prognosis
            # gene_annotation.best_gene_symbol_drug(best_gene_set)  # drug
            # gene_annotation.best_gene_symbol_diagnosis(best_gene_set)  # diagnosis
###############################################################################

    print('Number of deleted classifiers mcc < 0 : ', i, 'delete',i_, 'obtain')
    vector = numpy.array(centroid_vector_[value], dtype=object)
    plist = gene_partition_list[value]

    return vector, plist



# Calculate sampling probability based on the proportion of positive and negative samples
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
    # print(pro)  # At this point, the probability of minority class samples is higher

    # The following section is a substitute for the above section, which means that the probability of all samples being consistent
    # p = 1.0/l
    # pro = []
    # for i in range(0, l):
    #     pro.append(p)
    return pro


# Calculate sampling probability based on the proportion of positive and negative samples
def probability_weight(pro, y_ture, y_pred):
    num = 0
    for i in range(len(y_ture)):
        if y_ture[i] != y_pred[i]:
            num += 1
    err = num / len(y_ture)  # average error rate
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


# Calculate average and median values based on gene sets
def gene_set_statistics(gene_partition_list, express_data):
    statistics = []
    for gene_set in gene_partition_list:  # Traverse N gene sets, with M genes in each set
        # print(gene_set)
        gene_list = express_data[:, gene_set]  # Extract data from this gene set into a new array
        # print(gene_list.shape)
        # mean = numpy.mean(gene_list, axis=0)  # Compress rows and calculate the average number of each sample by column
        mean = numpy.mean(gene_list, axis=1)  # Compress columns and calculate the average number of each sample by row
        # median = numpy.median(gene_list, axis=0)  # Compress rows and calculate the median of each sample by column
        # print('mean:', mean.shape)
        statistics.append(numpy.array(mean))
        # statistics.append(numpy.array(median))
    statistics = numpy.array(statistics)
    # print(statistics.shape)
    return statistics.T
