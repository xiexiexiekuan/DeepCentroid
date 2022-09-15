import time
import scipy.io as scio
import numpy
import centroid
import partition
import drug_label
from sklearn.model_selection import StratifiedKFold

# 计时函数
start_time = time.time()
# 加载数据
print('数据预处理：')

# 癌症预后数据集--标准化
mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
express_data = numpy.array(mat['ma2'])  # 表达数据，病人-基因矩阵信息
prognosis = numpy.array(mat['co'][0])
gene_id = numpy.array(mat['id'])  # 基因编号
gene_number = express_data.shape[1]  # 基因数量
# 已知基因集合获取
gene_list_known_2 = partition.known_gene_set(gene_id, '../centroid_dataset/known_gene_set/c2.cp.kegg.v7.4.entrez.gmt')
print('已知基因集合c2数目：', gene_list_known_2.shape[0])
gene_list_known_5 = partition.known_gene_set(gene_id, '../centroid_dataset/known_gene_set/c5.go.bp.v7.4.entrez.gmt')
print('已知基因集合c5数目：', gene_list_known_5.shape[0])

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

# 输出数据信息
print('数据数据大小：', express_data.shape)
print('正样本数量：', sum(prognosis))
data_time = time.time()
print("数据预处理的时间：{:.5f} s.".format(data_time - start_time))

# 打印结果
mcc_set = []
auc_set = []

# # 5折交叉验证
# # print('数据空值数量：', numpy.isnan(express_data).sum())
# # bootstrap_size = 1  # 采样系数
# for k in range(0, 10):
#     print("第 {} 次交叉验证：".format(k+1))
#     # 按正负样本比例划分数据集
#     kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data, prognosis)  # 样本为行
#     i = 0
#     for train_, test_ in kf:
#         i = i + 1
#         print("第 {} 折：".format(i))
#         test_data = express_data[test_]
#         test_data_label = prognosis[test_]
#         train = express_data[train_]
#         train_label = prognosis[train_]
#
#         gene_partition_list = partition.random_cut_data(gene_number)  # 划分随机基因集合
#         gene_partition_list = numpy.append(gene_partition_list, gene_list_known_2, axis=0)  # 添加已知基因集合
#         gene_partition_list = numpy.append(gene_partition_list, gene_list_known_5, axis=0)
#         print('基因集合数目：', gene_partition_list.shape[0])
#
#         # 初始化
#         centroid.initialization()
#         # 训练模型
#         centroid.train(train, train_label, gene_partition_list)
#         # 预测
#         auc, mcc = centroid.predict(test_data, test_data_label)
#         mcc_set = numpy.append(mcc_set, mcc)
#         auc_set = numpy.append(auc_set, auc)


# 独立验证
for t in range(0, 10):
    gene_partition_list = partition.random_cut_data(gene_number)  # 划分随机基因集合
    gene_partition_list = numpy.append(gene_partition_list, gene_list_known_2, axis=0)  # 添加已知基因集合
    gene_partition_list = numpy.append(gene_partition_list, gene_list_known_5, axis=0)
    print('基因集合数目：', gene_partition_list.shape[0])

    # 初始化
    centroid.initialization()
    # 训练模型
    centroid.train(express_data, prognosis, gene_partition_list)
    # 测试数据加载
    mat_test = scio.loadmat('../centroid_dataset/prognosis_scale/test_data_top3.mat')
    data_test = numpy.array(mat_test['ma2'])
    prognosis_test = numpy.array(mat_test['co'][0])
    print('测试数据大小：', data_test.shape)
    # 预测
    auc, mcc = centroid.predict(data_test, prognosis_test)
    mcc_set = numpy.append(mcc_set, mcc)
    auc_set = numpy.append(auc_set, auc)

# 输出预测结果
print('结果 mcc ：')
for var in mcc_set:
    print(var)
print('结果 auc ：')
for var in auc_set:
    print(var)


# 程序结束信息
end_time = time.time()
# print("模型总用时：{:.2f} min.".format((end_time - data_time) / 60.0))
print("程序总用时：{:.2f} min.".format((end_time - start_time) / 60.0))
