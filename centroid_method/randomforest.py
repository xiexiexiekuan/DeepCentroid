import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


# 根据数据和标签自行划分测试集
def random_forest(dataset, labels):
    dataset = dataset.T
    labels = labels.T[0]
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, train_size=200)
    print('训练集：', x_train.shape)
    print('测试集：', x_test.shape)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("随机森林 acc: {:.3f}, mcc: {:.3f}".format(acc, mcc))


# 使用已有数据训练和预测
def random_forest_list(express_data_train, express_data_test, prognosis_train, prognosis_test):
    labels_train = prognosis_train
    # print('训练集标签：', labels_train)
    labels_test = prognosis_test
    # print('测试集标签: ', labels_test[0])
    data_train = express_data_train  # 行为样本，列为特征基因
    data_test = express_data_test

    model = RandomForestClassifier()
    model.fit(data_train, labels_train)
    y_pred = model.predict(data_test)
    y_pred_p = model.predict_proba(data_test)
    y_pred_p = numpy.array(y_pred_p[:, 1]).T
    acc = accuracy_score(labels_test, y_pred_p)
    mcc = matthews_corrcoef(labels_test, y_pred)
    print("随机森林 acc: {:.3f}, mcc: {:.3f}".format(acc, mcc))


# 根据训练数据生成一个随机森林
def random_forest_model(dataset, labels):
    model = RandomForestClassifier()
    model.fit(dataset, labels)

    return model


# 对每个模型，预测所有数据
def rf_predict_score(message, model, dataset, labels):
    r = []
    for item in model:
        y_pred_p = item.predict_proba(dataset)
        y_pred_p = numpy.array(y_pred_p[:, 1]).T
        r.append(y_pred_p)

    result_p = numpy.average(r, axis=0)
    result = numpy.int64(result_p >= 0.5)
    auc = roc_auc_score(labels, result_p)
    mcc = matthews_corrcoef(labels, result)

    print("随机森林{}集 auc: {:.3f}, mcc: {:.3f}".format(message, auc, mcc))
    return numpy.array(r).T, auc, mcc
