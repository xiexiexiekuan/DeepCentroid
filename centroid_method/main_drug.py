import time
import pandas
import scipy.io as scio
import numpy
import centroid
import partition
import drug_label
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import centroid_rf
import centroid_muti1

# Timing Function
start_time = time.time()

# Drug Sensitivity Prediction Dataset - Standardization
prognosis = drug_label.get_medicine_label()  # A set of labels for one behavior
medicine_num = prognosis.shape[0]  # Number of drugs, i.e. how many sets of labels
# gene expression data
# mat = scio.loadmat('../centroid_dataset/drug_sensitivity/express_data_scale.mat')
# express_data = numpy.array(mat['ma2'])
express_data = numpy.load('../centroid_dataset/drug_sensitivity/express_data_scale.npy')
# gene_number = express_data.shape[1]  # number of genes
print('Expression spectrum data:', express_data.shape)  # (949, 17419)
# Methylation data
# mat_m = scio.loadmat('../centroid_dataset/drug_sensitivity/methylation_data_scale.mat')
# express_data_m = numpy.array(mat_m['ma2'])  # Expression data, patient gene matrix information
express_data_m = numpy.load('../centroid_dataset/drug_sensitivity/methylation_data_scale.npy')
print('Methylation data:', express_data_m.shape)  # (949, 14726)
# gene_number_m = express_data_m.shape[1]  # number of genes
express_data = numpy.append(express_data, express_data_m, axis=1)  # Features of concatenating two types of data
gene_number = express_data.shape[1]  # number of genes

# Output data information
# print('Data size:', express_data.shape)
print('Number of drugs:', medicine_num)  # 48
data_time = time.time()
print("Data preprocessing time:{:.5f} s.".format(data_time - start_time))


# parameter
known_feature_set = False  # whether add a set of known genes or not. This project only provides known sets of prognostic data
is_feature_select = False  # Feature filtering or not
bootstrap_size = 0.65  # Sampling coefficient, each classifier divides the training set and validation set based on this ratio
gene_set_num = (int(express_data.shape[0]/50000) + 1)*500  # Number of feature sets
gene_set_min = 10  # The minimum value of the feature set range
gene_set_max = 200  # The maximum value of the feature set range
max_train_layer = 10  # Maximum number of training layers
cut_centroid = True  # Whether to prune or not
function_annotation = False  # Function annotation. After enabling, different annotation functions need to be manually selected based on the application scenario, located incentroid_compute.verify_centroid_distance
is_sliding_scan = False  # Whether to slide window scanning, default to random scanning

# Print Results
result_set = []

# 50 fold cross validation test for drug sensitivity, a total of 48 drugs
drug_list1 = [45, 40, 20, 23, 19, 21, 44, 16, 43, 46, 41, 22, 11]
drug_list2 = [14, 47, 39, 42, 13, 10, 15, 12, 18, 17, 32, 35, 36, 38, 25, 26]
drug_list3 = [28, 24, 29, 37, 27, 31, 30, 34, 33, 5, 4, 7, 8, 3, 2, 9, 6, 0, 1]

for k in drug_list3:
    # Reduce samples with unknown status based on labels
    express_data_, prognosis_ = drug_label.medicine_data_cut(express_data, prognosis[k])  # Obtain the characteristics and labels of the k-th drug
    print('Number of positive samples:', sum(prognosis_), ' / ', len(prognosis_))
    print("{} drug: {}".format(k, express_data_.shape))
    i = 0
    for j in range(0, 10):
        kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data_, prognosis_)  # Sample as row
        for train_, test_ in kf:
            i = i + 1
            layer_time = time.time()
            print("Fold {}:".format(i))
            test_data = express_data_[test_]
            test_data_label = prognosis_[test_]
            train = express_data_[train_]
            train_label = prognosis_[train_]

            smote = SMOTE()  # The proportion of drug categories is too large, and Oversampling is used
            train, train_label = smote.fit_resample(train, train_label)

            # initialization
            centroid_muti1.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
            # training model
            centroid_muti1.train(train, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)
            # prediction
            result_, l, p = centroid_muti1.predict(test_data, test_data_label)
            result_set.append(result_)
            print("the {}th fold is used: {:.2f} min.".format(i, (time.time() - layer_time) / 60.0))

print('accuracy, precision, recall, f1_score, auc, mcc')
print(result_set)
print(numpy.mean(result_set, axis=0))
data = pandas.DataFrame(result_set)
data.to_csv('result.csv', encoding='utf-8')

# Program End Information
end_time = time.time()
print("Total model time:{:.2f} min.".format((end_time - data_time) / 60.0))
# print("Total model time:{:.2f} min.".format((end_time - start_time) / 60.0))

