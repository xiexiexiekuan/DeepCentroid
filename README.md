# DeepCentroid
## brief introduction
Motivation: Cancer is a complex and diverse disease with extensive genetic heterogeneity. Even patients with the same type of cancer respond to anticancer drugs with wide variation, making precision medicine research a popular area in cancer research. Using omics data to construct classifiers for the corresponding prediction is a popular solution nowadays. However, the high feature dimensionality, small sample size, and category imbalance of omics data make it difficult to train effective and well-generalized models by traditional machine learning methods.  
Results: Combining the excellent stability of the centroid classifier and the strong fitting ability of the deep cascade strategy, we propose a new classifier, called DeepCentroid, and apply it to early diagnosis, prognosis, and drug sensitivity prediction of cancer. DeepCentroid is an integrated learning method with a multi-layer cascade structure. It is divided into two stages: feature scanning and cascade learning, and can dynamically adjust the training scale. Experimental results show that DeepCentroid achieves better performance than traditional machine learning models in all three problems mentioned above, confirming its promising application in precision medicine. In addition, functional annotations show that the features scanned by the model have biological significance, demonstrating that the model also possesses biological interpretability.
## Environmental configuration
Name Version, 
imblearn 0.7.12, 
libxslt 2.0.3, 
nltk 0.9.1, 
numpy 2.2.0, 
pandas 1.4.4, 
partd 0.26.2, 
python 4.2.1, 
scikit-learn 1.4.0, 
scipy 20.3.0, 
tqdm 1.5.6, 
## dataset
The experimental data of DeepCentroid are divided into three aspects: early cancer diagnosis aspects, cancer prognosis aspects and drug sensitivity prediction.  
The early cancer diagnosis data are plasma cell-free DNA whole genome sequencing data from lung cancer samples and controls.  
Cancer prognostic data were obtained from Gene Expression Omnibus, the training data were GSE2034 and the independent validation data were GSE7390 , GSE11121 and GSE12093 (Zhang et al. 2009). These four datasets include transcriptomic data from breast cancer patients, as well as clinical data such as follow-up information.  
Drug sensitivity prediction data were obtained from Genomics of Drug Sensitivity in Cancer, including DNA methylation data in multiple cell lines , Robust Multi-Array Average Normalized gene expression data, and drug sensitivity data for cell lines . A more specific presentation of the data can be found in the supplemental content.  
[You can download our data here](https://dmpmlab.github.io/Packages.html)
## Model structure
![image](https://github.com/xiexiexiekuan/DeepCentroid/assets/49866501/53691ce9-8fd1-4fa2-a4bd-24c1c2c98560)  
The DeepCentroid package provides a binary computational model for precision medicine that can integrate heterogeneous data.  
It is divided into two phases, feature scanning, and cascade learning, and has stable and efficient predictions in the fields of early diagnosis, prognosis estimation, and anti-cancer drug sensitivity of cancer.   
In the feature scanning stage, DeepCentroid can process heterogeneous information of different dimensions and dynamically divide the feature set according to the data size. In the cascade learning stage, DeepCentroid adopts a deeply cascaded strategy and integrates centroid classifiers to make predictions.
## User Manual
1. Download the project source code and data locally. There are three types of experimental data, among which early diagnosis and prognosis data also have independent validation data. Please pay attention to distinguishing.
2. Configure Python dependencies according to environmental requirements.
3. You can use the provided interface for personalized modifications.
-Selection of Datasets
-The size and quantity of gene sets
-Introduction of known gene sets
-Key gene result statistics
-Output level of model training
## Contact us
If you have any questions or needs regarding usage, you can edit the email to:1192450429@qq.com









