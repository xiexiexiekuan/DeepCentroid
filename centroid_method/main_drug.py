import time
import pandas
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

# 药物敏感性预测数据集--标准化
prognosis = drug_label.get_medicine_label()  # 一行为一组标签
medicine_num = prognosis.shape[0]  # 药物数量，即几组标签
# 基因表达数据
mat = scio.loadmat('../centroid_dataset/drug_sensitivity/express_data_scale.mat')
express_data = numpy.array(mat['ma2'])
# gene_number = express_data.shape[1]  # 基因数量
# 甲基化数据
mat_m = scio.loadmat('../centroid_dataset/drug_sensitivity/methylation_data_scale.mat')
express_data_m = numpy.array(mat_m['ma2'])  # 表达数据，病人-基因矩阵信息
# gene_number_m = express_data_m.shape[1]  # 基因数量
express_data = numpy.append(express_data, express_data_m, axis=1)
gene_number = express_data.shape[1]  # 基因数量

# 输出数据信息
print('数据数据大小：', express_data.shape)
data_time = time.time()
print("数据预处理的时间：{:.5f} s.".format(data_time - start_time))

# 打印结果
mcc_set = []
auc_set = []

# 5折交叉验证测试药物敏感性
drug_list1 = [0, 2, 4, 5]  # 测试8种药物
drug_list2 = [7, 9, 33, 38]  # 测试8种药物
result = []
result_ = []
for k in drug_list1:
    # 依据标签削减状态不明的样本
    express_data_, prognosis_ = drug_label.medicine_data_cut(express_data, prognosis[k])
    print('正样本数量：', sum(prognosis_))
    print("第 {} 种药物：{}".format(k, express_data_.shape))

    for j in range(0, 5):
        kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data_, prognosis_)  # 样本为行
        for train_, test_ in kf:

            test_data = express_data_[test_]
            test_data_label = prognosis_[test_]
            train = express_data_[train_]
            train_label = prognosis_[train_]

            gene_partition_list = partition.random_cut_data(gene_number)  # 划分随机基因集合
            print('基因集合数目：', gene_partition_list.shape[0])

            # 初始化
            centroid.initialization()
            # 训练模型
            centroid.train(train, train_label, gene_partition_list)
            # 预测
            auc, mcc = centroid.predict(test_data, test_data_label)
            result.append(auc)
            result_.append(mcc)

            # result = numpy.append(result, result_, axis=0)

# data = pandas.DataFrame(result)
# data.to_csv('result.csv', encoding='utf-8')
print("auc")
print(numpy.array(result))
print("mcc")
print(numpy.array(result_))

# 程序结束信息
end_time = time.time()
# print("模型总用时：{:.2f} min.".format((end_time - data_time) / 60.0))
print("程序总用时：{:.2f} min.".format((end_time - start_time) / 60.0))
