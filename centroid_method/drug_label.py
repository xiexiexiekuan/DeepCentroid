import csv
import numpy


# 药物敏感性预测当中获取数据标签
def get_medicine_label():
    data = csv.reader(open('../centroid_dataset/drug_sensitivity/medicine_label.csv', 'r'))
    cosmic_label = []
    for var in data:
        row = []
        for i in var:
            row.append(int(i))
        cosmic_label.append(row)
    cosmic_label = numpy.array(cosmic_label)
    return cosmic_label


# 在药物预测中，依据每种药物的标签删除掉敏感性不明确的细胞系
def medicine_data_cut(express_data, medicine_label):
    invalid_sample = numpy.where(medicine_label == -1)[0]
    express_data_ = numpy.delete(express_data, invalid_sample, axis=0)  # 表达数据，病人-基因矩阵信息
    prognosis = numpy.delete(medicine_label, invalid_sample)

    return express_data_, prognosis
