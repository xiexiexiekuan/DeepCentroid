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
from imblearn.over_sampling import SMOTE  # Random sampling function and SMOTE Oversampling function
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Timing Function
start_time = time.time()

# Cancer Prognosis Dataset - Standardization
mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
express_data = numpy.array(mat['ma2'])  # Expression data, patient gene matrix information
prognosis = numpy.array(mat['co'][0])
# gene_id = numpy.array(mat['id'])  # Gene IDs
gene_number = express_data.shape[1]  # Gene number
# Independent validation data
mat_test = scio.loadmat('../centroid_dataset/prognosis_scale/GSE7390.mat')
data_test = numpy.array(mat_test['ma2'])
prognosis_test = numpy.array(mat_test['co'][0])


# Output data information
print('data size:', express_data.shape, end="  ")
print('Number of positive samples:', sum(prognosis), end="  ")
print('Number of positive samples of test data:', sum(prognosis_test), end="  ")
print('Number of data blanks:', numpy.isnan(express_data).sum())
data_time = time.time()
print("Time for data preprocessing: {:.5f} s.".format(data_time - start_time))

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
# Please change the number of tests in the following code

# Print Results
result_set = []
l_set = []
p_set = []

# 5 cross validation
for k in range(0, 10):

    print("{} Cross validation:".format(k+1))
    # Divide the dataset according to the proportion of positive and negative samples
    kf = StratifiedKFold(n_splits=5, shuffle=True).split(express_data, prognosis)  # Sample as row
    i = 0
    for train_, test_ in kf:
        i = i + 1
        layer_time = time.time()
        print("Fold {}:".format(i))
        test_data = express_data[test_]
        test_data_label = prognosis[test_]
        train = express_data[train_]
        train_label = prognosis[train_]

        # initialization
        centroid.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
        # training model
        centroid.train(train, train_label, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)

        # prediction
        result_, l, p = centroid.predict(test_data, test_data_label)
        result_set.append(result_)  # Save 6 metrics  accuracy, precision, recall, f1_score, auc, mcc
        l_set.append(l)  # Save the label for the test set
        p_set.append(p)  # Save the prediction probability of the test set, which will be updated later with l_set merged into one file, draw AUC to use

        print("the {} th fold is used:{:.2f} min.".format(i, (time.time() - layer_time) / 60.0))


# Independent verification
for t in range(0, 1):
    layer_time = time.time()

    # initialization
    centroid.initialization(is_feature_select, gene_set_num, gene_set_min, gene_set_max)
    # training model
    centroid.train(express_data, prognosis, max_train_layer, known_feature_set, bootstrap_size, is_sliding_scan, cut_centroid, function_annotation)
    # Test data loading
    print('Test data size:', data_test.shape)
    # prediction: accuracy, precision, recall, f1_score, auc, mcc
    result_, l, p = centroid.predict(data_test, prognosis_test)
    result_set.append(result_)
    l_set.append(l)
    p_set.append(p)
    print("{} th usage time:{:.2f} min.".format(t, (time.time() - layer_time) / 60.0))

# Output prediction results
print('accuracy, precision, recall, f1_score, auc, mcc')
print(result_set)
print(numpy.mean(result_set, axis=0))
data = pandas.DataFrame(result_set)
data.to_csv('result_set.csv', encoding='utf-8')

for p in p_set:
    l_set.append(p)
data = pandas.DataFrame(l_set)
data = data.transpose()
data.to_csv('probability_set.csv', encoding='utf-8')

# Program End Information
end_time = time.time()
print("Total model time:{:.2f} min.".format((end_time - data_time) / 60.0))
# print("Total program time:{:.2f} min.".format((end_time - start_time) / 60.0))

