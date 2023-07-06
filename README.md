# DeepCentroid
**Deep** Cascade **Centroid** Classifier (DeepCentroid): A Robust Deep Cascade Classifier for Biomedical Omics Data Classification

## Table of Contents
1. [Quick start](#Quick-start)
2. [Usage](#Usage)
3. [Citation](#Citation)
4. [License](#License)
5. [Contact](#Contact)

## Quick start
1. Obtain files.  
	```
	git clone --recursive https://github.com/xiexiexiekuan/DeepCentroid.git
	cd centroid_method
	```
2. Extract data in folder ```centroid_dataset/prognosis_scale/```
3. Configure experimental environment. **This step can be skipped.**  
	```
	cd centroid_dataset
	conda env create -f=/path/py38.yaml -n name
	```
	* Windows
	* python 3.8 
	* numpy 2.2
	* scikit-learn 1.4
4. Run the example at your Python build tools such as Pycharm, or in bash command line.  
	```
	python main.py
	```  
5. You will receive the results of the test cases with six evaluation indicators.

## Usage

### We take 'Cancer prognisis prediction' as an example to show how to use our package.
Cancer prognostic data were obtained from Gene Expression Omnibus, the training data were GSE2034 and the independent validation data were GSE7390 , GSE11121 and GSE12093. These four datasets include transcriptomic data from breast cancer patients, as well as clinical data such as follow-up information.  
Before starting to run the code, you need to understand these parameters with default  values:  

* known_feature_set = False  # Whether to add a known gene set
* is_feature_select = False  # Whether to filter features
* bootstrap_size = 0.65  # sampling coefficient, each classifier divides the training set and validation set based on this ratio
* gene_set_num = (int(express_data.shape[0]/50000) + 1)*500  # Number of feature sets
* gene_set_min = 10  # The minimum value of the feature set range
* gene_set_max = 200  # The maximum value of the feature set range
* max_train_layer = 10  # Maximum number of training layers
* cut_centroid = True  # Whether to prune or not
* function_annotation = False  # Function annotation
* is_sliding_scan = False  # Whether to use sliding window scanning, default to random scanning

Description of dataset:  
* data feature: Two-dimensional data
* data label: One-dimensional data with 0 or 1

In the initial content of the code, the training set is GSE2034, and the test set is GSE7390. After independent verification once, you will receive an independent verification result file containing 6 indicators. 

### Add known gene sets
Make ```known_feature_set = True```
You can directly run the code, which will enable the known gene set of prognostic data.  
Add a new gene set:  
each line of the file represents a set, and the data in each set is separated by tabs. The content of the data should be consistent with the description of the original data, such as gene IDs.
#### Functional annotation
Make ```function_annotation = True```  
Then you will get the feature importance ranking file. If you need to correspond to the gene ID, you can see ```gene_annotation.py``` just for prognosis data.  

## Citation
```
Xie K & Zhou X. "DeepCentroid: A Robust Deep Cascade Classifier for Biomedical Omics Data Classification". In preparation.
```
## License
* For academic research, please refer to MIT license.
* For commerical usage, please contact the authors.

## Contact
* Kuan Xie <xiekuan@webmail.hzau.edu.cn>
* Xionghui Zhou <zhouxionghui@mail.hzau.edu.cn>
