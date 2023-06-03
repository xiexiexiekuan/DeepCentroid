import time
import pandas
import scipy.io as scio
import numpy
import centroid
import partition
import drug_label
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# 计时函数
start_time = time.time()
# 加载数据
print('数据预处理：')

# 药物敏感性预测数据集--标准化
prognosis = drug_label.get_medicine_label()  # 一行为一组标签
medicine_num = prognosis.shape[0]  # 药物数量，即几组标签
# 基因表达数据
# mat = scio.loadmat('../centroid_dataset/drug_sensitivity/express_data_scale.mat')
# express_data = numpy.array(mat['ma2'])
express_data = numpy.load('../centroid_dataset/drug_sensitivity/express_data_scale.npy')
# gene_number = express_data.shape[1]  # 基因数量
print('表达谱数据：', express_data.shape)  # (949, 17419)
# 甲基化数据
# mat_m = scio.loadmat('../centroid_dataset/drug_sensitivity/methylation_data_scale.mat')
# express_data_m = numpy.array(mat_m['ma2'])  # 表达数据，病人-基因矩阵信息
express_data_m = numpy.load('../centroid_dataset/drug_sensitivity/methylation_data_scale.npy')
print('甲基化数据：', express_data_m.shape)  # (949, 14726)
# gene_number_m = express_data_m.shape[1]  # 基因数量
express_data = numpy.append(express_data, express_data_m, axis=1)  # 拼接两种数据的特征
gene_number = express_data.shape[1]  # 基因数量

# 输出数据信息
# print('数据数据大小：', express_data.shape)
print('药物数量：', medicine_num)  # 48
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
cut_centroid = True  # 是否剪枝
function_annotation = False  # 功能注释，启用后需要手动根据应用场景选择不同的注释函数，位置在centroid_compute.verify_centroid_distance
is_sliding_scan = False  # 是否滑动窗口扫描，默认为随机扫描

# 打印结果
result_set = []

# 5折交叉验证测试药物敏感性，一共48个药物
# drug_list = [0, 2, 4, 5, 7, 9, 33, 38]  # 测试8种药物
drug_list1 = [45, 40, 20, 23, 19, 21, 44, 16]
drug_list2 = [43, 46, 41, 22, 11, 14, 47, 39, 42, 13]
drug_list3 = [10, 15, 12, 18, 17, 32, 35, 36, 38, 25]
drug_list4 = [26, 28, 24, 2, 37, 27, 31, 30, 34, 33]
drug_list5 = [5, 4, 7, 8, 3, 2, 9, 6, 0, 1]
drug_list6 = [0]

for k in drug_list6:
    # 依据标签削减状态不明的样本
    express_data_, prognosis_ = drug_label.medicine_data_cut(express_data, prognosis[k])  # 获取第k个药物的特征以及标签
    print('正样本数量：', sum(prognosis_), ' / ', len(prognosis_))
    print("第 {} 种药物：{}".format(k, express_data_.shape))
    i = 0
    for j in range(0, 1):
        kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data_, prognosis_)  # 样本为行
        for train_, test_ in kf:
            i = i + 1
            layer_time = time.time()
            print("第 {} 折：".format(i))
            test_data = express_data_[test_]
            test_data_label = prognosis_[test_]
            train = express_data_[train_]
            train_label = prognosis_[train_]

            # smote = SMOTE()  # 药物类别比例过大，使用过采样
            # train, train_label = smote.fit_resample(train, train_label)

            # 初始化
            centroid.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
            # 训练模型
            centroid.train(train, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)
            # 预测
            result_, l, p = centroid.predict(test_data, test_data_label)
            result_set.append(result_)
            print("第 {} 折用时：{:.2f} min.".format(i, (time.time() - layer_time) / 60.0))

print('accuracy, precision, recall, f1_score, auc, mcc')
print(result_set)
print(numpy.mean(result_set, axis=0))
# data = pandas.DataFrame(result_set)
# data.to_csv('result.csv', encoding='utf-8')

# 程序结束信息
end_time = time.time()
print("模型总用时：{:.2f} min.".format((end_time - data_time) / 60.0))
# print("程序总用时：{:.2f} min.".format((end_time - start_time) / 60.0))

