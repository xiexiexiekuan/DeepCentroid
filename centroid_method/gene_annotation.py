import time
import pandas
import scipy.io as scio
import numpy
import csv



def best_gene_symbol_prognosis(dic):
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(dic_sort)

    # Cancer Prognosis Dataset - Standardization
    mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
    gene_id = numpy.array(mat['id'])[0]  # Gene number


    best_set = []
    for i in dic_sort:

        s = [gene_id[i[0]],i[1]]
        best_set.append(s)

    # print(best_set)
    data = pandas.DataFrame(best_set)
    data.to_csv('best_set.csv', encoding='utf-8')