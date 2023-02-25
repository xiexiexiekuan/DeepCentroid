# import time
# import pandas
# import centroid_compute
# import numpy
# from sklearn import metrics
# import partition
# import simple_centroid
# from centroid_compute import threshold
# from sklearn.preprocessing import scale, normalize, minmax_scale
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from deepforest import CascadeForestClassifier
# import xgboost
# from sklearn import svm
#
# centroid_vector = []  # 第一层质心分类器生成的质心向量
# gene_partition_list = []  # 质心分类器基因集合
# level_data = []  # 层之间传输数据
# layer = 1  # 训练层数
# valid_layer = 1  # 有效训练层数
# probability = []
# model = []
#
#
#
# # 初始化
# def initialization():
#     global centroid_vector, gene_partition_list, level_data, layer, valid_layer, model
#     centroid_vector = []
#     gene_partition_list = []
#     level_data = []
#     layer = 1
#     valid_layer = 1
#     model = []
#
#
# # 测试集计算ACC, AUC, MCC
# def model_score(y_ture, distance, sign=False):
#
#     if sign:
#         y_pred_p = numpy.int64(distance >= 0)  # 预测概率
#         y_pred_p = numpy.average(y_pred_p, axis=0)
#         y_pred = numpy.int64(y_pred_p >= 0.5)  # 预测标签
#     else:
#         y_pred_p = numpy.int64(distance >= 0.5)  # 预测概率
#         y_pred_p = numpy.average(y_pred_p, axis=0)
#         y_pred = numpy.int64(y_pred_p >= 0.5)  # 预测标签
#
#     accuracy = metrics.accuracy_score(y_ture, y_pred)  # 真实值，预测值
#     precision = metrics.precision_score(y_ture, y_pred)
#     recall = metrics.recall_score(y_ture, y_pred)
#     f1_score = metrics.f1_score(y_ture, y_pred)
#     auc = metrics.roc_auc_score(y_ture, y_pred_p)
#     mcc = metrics.matthews_corrcoef(y_ture, y_pred)
#
#     result_set = numpy.array([accuracy, precision, recall, f1_score, auc, mcc])
#
#     if sign:
#         return result_set, y_pred
#     else:
#         return result_set, y_pred_p
#
#
# # 训练
# def train(train_data_, train_label, max_train_layer=10, bootstrap_size=1):
#     global layer, valid_layer, centroid_vector, gene_partition_list, level_data, probability
#
#     # 获取每个基因集合对应的训练集，训练标签，验证集，验证标签
#     probability = centroid_compute.probability(train_label)
#     mcc = 0
#     express_data = train_data_
#
#     print('第 {} 层训练：'.format(layer), end=" ")
#     layer_time = time.time()  # 本层计算时间
#     print("输入数据：", express_data.shape)
#     # 基因集合数目
#     partition_list = partition.random_cut_data(express_data.shape[1])  # 划分随机基因集合
#     partition_num = partition_list.shape[0]
#     data_cut_set = centroid_compute.data_divide(partition_num, bootstrap_size, partition_list, express_data, train_label, probability)
#
#     # 计算所有基因的质心向量，保存输出的N*M*2的矩阵，即每个基因集合计算出关于质心的M*2个变量
#     vector_ = centroid_compute.centroid_vector(partition_num, data_cut_set)
#
#     # 根据验证集verify结果剔除不好的分类器，只在第一层
#     centroid_vector, gene_partition_list, err_set = centroid_compute.verify_centroid_distance(partition_list, vector_, data_cut_set)
#
#     # 计算训练集与质心的距离用于下一层训练
#     distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, express_data)
#     # 保留质心距离用于下层训练
#     level_data = numpy.array(distance.T).astype(numpy.float32)
#     result, y_pred = model_score(train_label, distance, True)
#     print("训练集-质心-预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(result[0], result[4], result[5]), end="  ")
#     print("time：{:.2f} min.".format((time.time() - layer_time) / 60.0))
#
#     # 集成多种分类器
#     m = RandomForestClassifier()
#     m.fit(level_data, train_label)
#     model.append(m)
#
#     m = svm.SVC(kernel='rbf', C=10, probability=True)
#     m.fit(level_data, train_label)
#     model.append(m)
#
#     m = CascadeForestClassifier(verbose=0)
#     m.fit(level_data, train_label)
#     model.append(m)
#
#     dtrain = xgboost.DMatrix(level_data, label=train_label)
#     param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic',
#              'eval_metric': ['logloss', 'auc', 'error']}  # 设置XGB的参数，使用字典形式传入
#     bst = xgboost.train(param, dtrain=dtrain)  # 训练
#     model.append(bst)
#
#     m = MLPClassifier(solver='lbfgs')
#     m.fit(level_data, train_label)
#     model.append(m)
#
#     print('集成训练完毕')
#
#
# # 预测
# def predict(test_data, test_data_label):
#
#     c_distance = test_data
#     c_distance = centroid_compute.sample_centroid_distance(gene_partition_list, centroid_vector, c_distance)
#     result, y_pred_p = model_score(test_data_label, c_distance, True)
#     print("{} 层测试集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(1, result[0], result[4], result[5]))
#     c_distance = numpy.array(c_distance.T).astype(numpy.float32)
#
#     result = []
#     for i in range(0, 5):
#         m = model[i]
#         if i == 3:
#             dtest = xgboost.DMatrix(c_distance)
#             y_pred_p = m.predict(dtest)  # 预测
#             result.append(y_pred_p)
#         else:
#             y_pred_p = m.predict_proba(c_distance)
#             y_pred_p = y_pred_p[:, 1]
#             y_pred_p = numpy.array(y_pred_p).T
#             result.append(y_pred_p)
#
#
#     result = numpy.array(result)
#     result, y_pred_p = model_score(test_data_label, result)
#     print("{} 层测试集预测结果ACC = {:.5f}, AUC = {:.5f}, MCC = {:.5f}".format(2, result[0], result[4], result[5]))
#
#     return result, test_data_label, y_pred_p
