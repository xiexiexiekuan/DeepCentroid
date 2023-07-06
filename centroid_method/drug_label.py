import csv
import numpy


# Obtaining data labels in drug sensitivity prediction
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


# In drug prediction, remove cell lines with unclear sensitivity based on the label of each drug
def medicine_data_cut(express_data, medicine_label):
    invalid_sample = numpy.where(medicine_label == -1)[0]
    express_data_ = numpy.delete(express_data, invalid_sample, axis=0)  # Expression data, patient gene matrix information

    prognosis = numpy.delete(medicine_label, invalid_sample)

    return express_data_, prognosis
