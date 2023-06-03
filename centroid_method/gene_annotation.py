import time
import pandas
import scipy.io as scio
import numpy
import csv



def best_gene_symbol_prognosis(dic):
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(dic_sort)

    # 癌症预后数据集--标准化
    mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
    gene_id = numpy.array(mat['id'])[0]  # 基因编号


    best_set = []
    for i in dic_sort:

        s = [gene_id[i[0]],i[1]]
        best_set.append(s)

    # print(best_set)
    data = pandas.DataFrame(best_set)
    data.to_csv('best_set.csv', encoding='utf-8')


def best_gene_symbol_drug(dic):
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(dic_sort)

    # 表达谱特征-基因符号
    gene_symbol = numpy.load('../centroid_dataset/drug_sensitivity/express_gene_symbol.npy')
    # print(gene_symbol)
    # 甲基化特征-染色体片段
    fragment = csv.reader(open('../centroid_dataset/drug_sensitivity/methylation_feature.csv', 'r'))
    gene_fragment = []
    for var in fragment:
        row = []
        for i in var:
            row.append(str(i))
        gene_fragment.append(row)
    gene_fragment = numpy.array(gene_fragment)
    # print(gene_fragment)

    best_set_symbol = []
    best_set_fragment = []
    for i in dic_sort:
        if i[0] >= 17419:  # 此时的特征为甲基化的
            k = i[0] - 17419
            u = gene_fragment[k]
            r = []
            for m in u:
                r.append(m)
            r.append(i[1])
            best_set_fragment.append(r)
        else:
            s = [gene_symbol[i[0]],i[1]]
            best_set_symbol.append(s)

    # print(best_set_fragment)
    # print('===========')
    # print(best_set_symbol)
    data = pandas.DataFrame(best_set_symbol)
    data.to_csv('best_set_symbol.csv', encoding='utf-8')
    data = pandas.DataFrame(best_set_fragment)
    data.to_csv('best_set_fragment.csv', encoding='utf-8')


def best_gene_symbol_diagnosis(dic):
    dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(dic_sort)

    # ENSG ID与基因名称转化
    # gencode_ = numpy.load('../centroid_dataset/liver_cancer/gencode.npy')
    gencode = {}
    # 肝癌的特征与肺癌的特征对应关系一致
    data = open('../centroid_dataset/liver_cancer/fragment_feature/mart_export.txt')
    for i in data:
        l = len(i)
        d = i[0:l - 1]
        d1 = d.split('\t')
        if d1[1] == '':
            continue
        gencode[d1[0]] = d1[1]

    # for i in gencode_:
    #     gencode[i[0]] = i[1]
    # print(gencode)

    # 特征对应的ENSG ID
    tss_feature = numpy.load('../centroid_dataset/liver_cancer/IFS_ENSG_feature.npy')
    # print(tss_feature)

    for i in dic_sort:
        print(i[1])
        break
    best_set = []
    for i in dic_sort:
        ensg = tss_feature[i[0]]
        if ensg in gencode:
            s = [gencode[ensg],i[1]]
            best_set.append(s)
        else:
            print(ensg, i[1])

    # print(best_set)
    data = pandas.DataFrame(best_set)
    data.to_csv('best_set.csv', encoding='utf-8')