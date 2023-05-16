import numpy
from sklearn import preprocessing
# prognosis_ = numpy.load('../centroid_dataset/liver_cancer/label_train_.npy')
# print(prognosis_)
# prognosis = numpy.load('../centroid_dataset/liver_cancer/label_train.npy')
# print(prognosis)


# data = open('../centroid_dataset/liver_cancer/fragment_feature/Homo_sapiens.GRCh38.109.chr.gtf')
# j=0
# for i in data:
#     print(i)
#     j+=1
#     if j == 50:
#         break


# data = open('../centroid_dataset/liver_cancer/fragment_feature/mart_export.txt')
# j=0
# for i in data:
#     l = len(i)
#     d = i[0:l-1]
#     d1 = d.split('\t')
#     if d1[1]=='':
#         continue
#     print(d1)
#     j+=1
#     if j == 50:
#         break


# ENSG00000210049	MT-TF
# ENSG00000211459	MT-RNR1
# ENSG00000210077	MT-TV
# ENSG00000210082	MT-RNR2
# ENSG00000209082	MT-TL1
# ENSG00000198888	MT-ND1
# ENSG00000210100	MT-TI
# ENSG00000210107	MT-TQ
# ENSG00000210112	MT-TM


data1 = numpy.load('../centroid_dataset/response/response_gide.npy')
label1 = numpy.load('../centroid_dataset/response/response_gide_label.npy')
data2 = numpy.load('../centroid_dataset/response/response_kim.npy')
label2 = numpy.load('../centroid_dataset/response/response_kim_label.npy')
data3 = numpy.load('../centroid_dataset/response/response_liu.npy')
label3 = numpy.load('../centroid_dataset/response/response_liu_label.npy')
data4 = numpy.load('../centroid_dataset/response/response_riaz.npy')
label4 = numpy.load('../centroid_dataset/response/response_riaz_label.npy')
# express_data = preprocessing.scale(data3,axis=1)
# prognosis = label3
# data_test = numpy.append(data1, data2, axis=0)
# data_test = numpy.append(data_test, data4, axis=0)
# prognosis_test = numpy.append(label1, label2)
# prognosis_test = numpy.append(prognosis_test, label4)
# print(data_test.shape)
# print(len(prognosis_test))
print(sum(label1),'/', len(label1))
print(sum(label2),'/', len(label2))
print(sum(label3),'/', len(label3))
print(sum(label4),'/', len(label4))
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data3)
data3 = preprocessing.scale(data3,axis = 1)
print(data3)