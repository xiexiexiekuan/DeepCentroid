import time
import pandas
import scipy.io as scio
import numpy
import centroid
import centroid_rf
import centroid_muti
import drug_label
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE  # 随机采样函数 和SMOTE过采样函数
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 计时函数
start_time = time.time()

# 加载数据
# 癌症预后数据集--标准化
# mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
# express_data = numpy.array(mat['ma2'])  # 表达数据，病人-基因矩阵信息
# prognosis = numpy.array(mat['co'][0])
# # gene_id = numpy.array(mat['id'])  # 基因编号
# gene_number = express_data.shape[1]  # 基因数量
# # 独立验证数据
# mat_test = scio.loadmat('../centroid_dataset/prognosis_scale/test_data_top3.mat')
# data_test = numpy.array(mat_test['ma2'])
# prognosis_test = numpy.array(mat_test['co'][0])

# # 癌症早期诊断数据集--标准化
# mat = scio.loadmat('../centroid_dataset/early_diagnosis.mat')
# express_data = numpy.array(mat['ma2'])  # 表达数据，病人-基因矩阵信息
# prognosis = numpy.array(mat['co'])[0]  # 样本编号
# gene_number = express_data.shape[1]  # 基因数量

# # 药物敏感性预测数据集--标准化
# prognosis = drug_label.get_medicine_label()  # 一行为一组标签
# medicine_num = prognosis.shape[0]  # 药物数量，即几组标签
# # 基因表达数据
# mat = scio.loadmat('../centroid_dataset/drug_sensitivity/express_data_scale.mat')
# express_data = numpy.array(mat['ma2'])
# # gene_number = express_data.shape[1]  # 基因数量
# # 甲基化数据
# mat_m = scio.loadmat('../centroid_dataset/drug_sensitivity/methylation_data_scale.mat')
# express_data_m = numpy.array(mat_m['ma2'])  # 表达数据，病人-基因矩阵信息
# # gene_number_m = express_data_m.shape[1]  # 基因数量
# express_data = numpy.append(express_data, express_data_m, axis=1)
# gene_number = express_data.shape[1]  # 基因数量

# # IFS乳腺癌数据--交叉验证数据
# ifs = numpy.load('../centroid_dataset/nature_IFS/IFS.npy')
# ifs = ifs.T
# ifs1 = ifs[24:78]
# patient = numpy.ones(ifs1.shape[0], dtype=int)
# ifs2 = numpy.append(ifs[133:347], ifs[421:422], axis=0)
# normal = numpy.zeros(ifs2.shape[0], dtype=int)
# prognosis = numpy.append(patient, normal)
# express_data = numpy.append(ifs1, ifs2, axis=0)
# express_data = preprocessing.scale(express_data, axis=1)
# gene_number = express_data.shape[1]  # 基因数量
# # 独立验证数据
# ifs = numpy.load('../centroid_dataset/nature_IFS/cc_breast_IFS.npy')
# patient = numpy.ones(25, dtype=int)
# normal = numpy.zeros(25, dtype=int)
# prognosis_test = numpy.append(patient, normal)
# data_test = ifs.T
# data_test = preprocessing.scale(data_test, axis=1)

# express_data = numpy.load('../centroid_dataset/nature_IFS/combat/ifs_combat_cut_train.npy')
# express_data = preprocessing.scale(express_data, axis=1)
# prognosis = numpy.load('../centroid_dataset/nature_IFS/ifs_label_train.npy')
# data_test = numpy.load('../centroid_dataset/nature_IFS/combat/ifs_combat_cut_test.npy')
# data_test = preprocessing.scale(data_test, axis=1)
# prognosis_test = numpy.load('../centroid_dataset/nature_IFS/ifs_label_test.npy')
# express_data = numpy.load('../centroid_dataset/nature_IFS/cut-0/ifs-0_train.npy')
# express_data = preprocessing.scale(express_data, axis=1)
# prognosis = numpy.load('../centroid_dataset/nature_IFS/ifs_label_train.npy')
# data_test = numpy.load('../centroid_dataset/nature_IFS/cut-0/ifs-0_test.npy')
# data_test = preprocessing.scale(data_test, axis=1)
# prognosis_test = numpy.load('../centroid_dataset/nature_IFS/ifs_label_test.npy')

# # 肝癌
express_data = numpy.load('../centroid_dataset/liver_cancer/IFS_train_.npy')
prognosis = numpy.load('../centroid_dataset/liver_cancer/label_train_.npy')
data_test = numpy.load('../centroid_dataset/liver_cancer/IFS_.npy')
prognosis_test = numpy.load('../centroid_dataset/liver_cancer/label_.npy')


# delfei早期诊断
# express_data = numpy.load('../centroid_dataset/delfei_early/delfei_train_data.npy')
# # express_data = preprocessing.scale(express_data, axis=1)
# prognosis = numpy.load('../centroid_dataset/delfei_early/delfei_train_label.npy')
# data_test = numpy.load('../centroid_dataset/delfei_early/delfei_test_data.npy')
# # data_test = preprocessing.scale(data_test, axis=1)
# prognosis_test = numpy.load('../centroid_dataset/delfei_early/delfei_test_label.npy')

# data_test = numpy.load('../centroid_dataset/delfei_early/delfei_train_data.npy')
# prognosis_test = numpy.load('../centroid_dataset/delfei_early/delfei_train_label.npy')
# express_data = numpy.load('../centroid_dataset/delfei_early/delfei_test_data.npy')
# prognosis = numpy.load('../centroid_dataset/delfei_early/delfei_test_label.npy')


# 输出数据信息
print('数据大小：', express_data.shape, end="  ")
print('正样本数量：', sum(prognosis), end="  ")
print('数据空值数量：', numpy.isnan(express_data).sum())
data_time = time.time()
print("数据预处理的时间：{:.5f} s.".format(data_time - start_time))

# 打印结果
result_set = []
l_set = []
p_set = []
bootstrap_size = 0.9  # 采样系数

# # 5折交叉验证
# for k in range(0, 1):
#
#     print("第 {} 次交叉验证：".format(k+1))
#     # 按正负样本比例划分数据集
#     kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data, prognosis)  # 样本为行
#     i = 0
#     for train_, test_ in kf:
#         i = i + 1
#         layer_time = time.time()
#         print("第 {} 折：".format(i))
#         test_data = express_data[test_]
#         test_data_label = prognosis[test_]
#         train = express_data[train_]
#         train_label = prognosis[train_]
#
#         # smote = SMOTE()
#         # train, train_label = smote.fit_resample(train, train_label)  # 使用原始数据的特征变量和目标变量生成过采样数据集
#
#         # 初始化
#         centroid.initialization()
#         # 训练模型
#         centroid.train(train, train_label, bootstrap_size=bootstrap_size)
#
#         # 预测 accuracy, precision, recall, f1_score, auc, mcc
#         result_, l, p = centroid.predict(test_data, test_data_label)
#         result_set.append(result_)
#         l_set.append(l)
#         p_set.append(p)
#
#         print("第 {} 折用时：{:.2f} min.".format(i, (time.time() - layer_time) / 60.0))


# 独立验证
for t in range(0, 5):
    layer_time = time.time()

    # data = SelectFromModel(RandomForestClassifier()).fit(express_data, prognosis)
    # print(data.get_support())
    # print(data.transform(express_data).shape)
    # express_data = express_data[:, data.get_support()]
    # print(express_data.shape)


    # smote = SMOTE()
    # express_data, prognosis = smote.fit_resample(express_data, prognosis)  # 使用原始数据的特征变量和目标变量生成过采样数据集

    # 初始化
    centroid.initialization()
    # 训练模型
    centroid.train(express_data, prognosis, bootstrap_size=bootstrap_size)
    # 测试数据加载

    print('测试数据大小：', data_test.shape)
    # 预测 accuracy, precision, recall, f1_score, auc, mcc
    result_, l, p = centroid.predict(data_test, prognosis_test)
    result_set.append(result_)
    l_set.append(l)
    p_set.append(p)
    print("第 {} 次用时：{:.2f} min.".format(t, (time.time() - layer_time) / 60.0))

# 输出预测结果
print('accuracy, precision, recall, f1_score, auc, mcc')
print(result_set)
print(numpy.mean(result_set, axis=0))
# data = pandas.DataFrame(result_set)
# data.to_csv('result_set.csv', encoding='utf-8')
#
# for p in p_set:
#     l_set.append(p)
# data = pandas.DataFrame(l_set)
# data = data.transpose()
# data.to_csv('l_set.csv', encoding='utf-8')

# 程序结束信息
end_time = time.time()
print("模型总用时：{:.2f} min.".format((end_time - data_time) / 60.0))
# print("程序总用时：{:.2f} min.".format((end_time - start_time) / 60.0))

