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
# 癌症预后数据集--标准化--使用中
mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
express_data = numpy.array(mat['ma2'])  # 表达数据，病人-基因矩阵信息
prognosis = numpy.array(mat['co'][0])
# gene_id = numpy.array(mat['id'])  # 基因编号
gene_number = express_data.shape[1]  # 基因数量
# 独立验证数据
mat_test = scio.loadmat('../centroid_dataset/prognosis_scale/test_data_top3.mat')
data_test = numpy.array(mat_test['ma2'])
prognosis_test = numpy.array(mat_test['co'][0])

# # 肺癌早期诊断--使用中
# express_data = numpy.load('../centroid_dataset/data20230315/lung/lung_train/coverage.npy')
# express_data = express_data.T
# express_data = preprocessing.scale(express_data, axis=1)
# prognosis = numpy.load('../centroid_dataset/data20230315/lung/ega_train_id.npy')
# data_test = numpy.load('../centroid_dataset/data20230315/lung/lung_test/coverage.npy')
# data_test = data_test.T
# data_test = preprocessing.scale(data_test, axis=1)
# prognosis_test = numpy.load('../centroid_dataset/data20230315/lung/ega_test_id.npy')
# 跟下面那个是一个源数据

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

# # 癌症早期诊断数据集--标准化
# mat = scio.loadmat('../centroid_dataset/early_diagnosis.mat')
# express_data = numpy.array(mat['ma2'])  # 表达数据，病人-基因矩阵信息
# prognosis = numpy.array(mat['co'])[0]  # 样本编号
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
# express_data = numpy.load('../centroid_dataset/liver_cancer/IFS_train_.npy')
# prognosis = numpy.load('../centroid_dataset/liver_cancer/label_train_.npy')
# data_test = numpy.load('../centroid_dataset/liver_cancer/IFS_.npy')
# prognosis_test = numpy.load('../centroid_dataset/liver_cancer/label_.npy')

# # drug_response
# data1 = numpy.load('../centroid_dataset/response/response_gide.npy')
# label1 = numpy.load('../centroid_dataset/response/response_gide_label.npy')
# data2 = numpy.load('../centroid_dataset/response/response_kim.npy')
# label2 = numpy.load('../centroid_dataset/response/response_kim_label.npy')
# data3 = numpy.load('../centroid_dataset/response/response_liu.npy')
# label3 = numpy.load('../centroid_dataset/response/response_liu_label.npy')
# data4 = numpy.load('../centroid_dataset/response/response_riaz.npy')
# label4 = numpy.load('../centroid_dataset/response/response_riaz_label.npy')
# # express_data = numpy.append(data3, data4, axis=0)
# # # # # express_data = preprocessing.scale(express_data, axis=1)
# # prognosis = numpy.append(label3, label4)
# # data_test = numpy.append(data1, data2, axis=0)
# # data_test = numpy.append(data_test, data4, axis=0)
# # # data_test = preprocessing.scale(data_test,axis=1)
# # prognosis_test = numpy.append(label1, label2)
# # prognosis_test = numpy.append(prognosis_test, label4)
# # data_test = numpy.append(data4, data1, axis=0)
# # data_test = preprocessing.scale(data_test, axis=1)
# # prognosis_test = numpy.append(label4, label1)
# express_data = data3
# prognosis = label3
# data_test = numpy.append(data1, data2, axis=0)
# prognosis_test = numpy.append(label1, label2)
#
# express_data = preprocessing.scale(express_data, axis=1)
# data_test = preprocessing.scale(data_test, axis=1)


# 输出数据信息
print('数据大小：', express_data.shape, end="  ")
print('正样本数量：', sum(prognosis), end="  ")
print('数据空值数量：', numpy.isnan(express_data).sum())
data_time = time.time()
print("数据预处理的时间：{:.5f} s.".format(data_time - start_time))

# 参数
known_feature_set = False  # 是否添加已知基因集合，本项目仅提供预后数据已知集合
is_feature_select = False  # 是否特征筛选
bootstrap_size = 0.65  # 采样系数，每个分类器依据该比例划分训练集与验证集
gene_set_num = (int(express_data.shape[0]/50000) + 1)*500  # 特征集合数
gene_set_num = 100  # 便捷修改处
gene_set_min = 10  # 特征集合范围的最小值
gene_set_max = 20  # 特征集合范围的最大值
max_train_layer = 10  # 最大训练层数
cut_centroid = False  # 是否剪枝
function_annotation = False  # 功能注释，启用后需要手动根据应用场景选择不同的注释函数，位置在centroid_compute.verify_centroid_distance
is_sliding_scan = False  # 是否滑动窗口扫描，默认为随机扫描
# 测试次数请在下面代码中更改

# 打印结果
result_set = []
l_set = []
p_set = []

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
#         # 初始化
#         centroid.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
#         # 训练模型
#         centroid.train(train, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)
#
#         # 预测
#         result_, l, p = centroid.predict(test_data, test_data_label)
#         result_set.append(result_)  # 保存6个指标  accuracy, precision, recall, f1_score, auc, mcc
#         l_set.append(l)  # 保存测试集的标签
#         p_set.append(p)  # 保存测试集的预测概率，后期会与l_set合并为一个文件，画AUC使用
#
#         print("第 {} 折用时：{:.2f} min.".format(i, (time.time() - layer_time) / 60.0))


# 独立验证
for t in range(0, 2):
    layer_time = time.time()

    # 初始化
    centroid.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
    # 训练模型
    centroid.train(express_data, prognosis, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)
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

