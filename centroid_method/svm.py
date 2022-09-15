# from sklearn import svm
# from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
#
#
# def svm_list(express_data_train, express_data_test, prognosis_train, prognosis_test):
#     classifier = svm.SVC(C=10, probability=True)  # C=2,kernel='rbf',gamma=10,decision_function_shape='ovr'  ovr:一对多策略
#     classifier.fit(express_data_train, prognosis_train)
#     y_pred = classifier.predict(express_data_test)
#     acc = accuracy_score(prognosis_test, y_pred)
#     mcc = matthews_corrcoef(prognosis_test, y_pred)
#     print("SVM   acc: {:.3f}, mcc: {:.3f}".format(acc, mcc))
#     return mcc
#
#
# def svm_model(dataset, labels):
#     model = svm.SVC(C=10, probability=True)
#     model.fit(dataset, labels)
#     y_pred = model.predict(dataset)
#     auc = roc_auc_score(labels, y_pred)
#     mcc = matthews_corrcoef(labels, y_pred)
#     print("SVM 训练集 auc: {:.3f}, mcc: {:.3f}".format(auc, mcc))
#     return model
#
#
# def svm_model_predict(model, dataset, labels):
#     y_pred = model.predict(dataset)
#     auc = roc_auc_score(labels, y_pred)
#     mcc = matthews_corrcoef(labels, y_pred)
#     print("SVM 测试集 auc: {:.3f}, mcc: {:.3f}".format(auc, mcc))
#     return auc, mcc
