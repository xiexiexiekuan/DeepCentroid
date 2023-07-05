# DeepCentroid
## Abstract
**Motivation:** Classification of samples using biomedical omics data is a widely employed method in biomedical research. However, these datasets often possess challenging characteristics, including high dimensionality, limited sample sizes, imbalanced class distributions, and inherent biases across diverse sources. These factors limit the performance of traditional machine learning models, particularly when applied to independent datasets.  
**Results:** To address these challenges, we propose a novel classifier, DeepCentroid, which combines the stability of the centroid classifier and the strong fitting ability of the deep cascade strategy. DeepCentroid is an ensemble learning method with a multi-layer cascade structure, consisting of feature scanning and cascade learning stages that can dynamically adjust the training scale. We apply DeepCentroid to three precision medicine applications—early diagnosis, prognosis, and drug sensitivity prediction of cancer—using cell-free DNA fragmentations, gene expression profiles, and DNA methylation data. Experimental results demonstrate that DeepCentroid outperforms six traditional machine learning models in all three applications, showcasing its potential in biological omics data classification. Furthermore, functional annotations reveal that the features scanned by the model exhibit biological significance, indicating its interpretability from a biological perspective. Our findings underscore the promising application of DeepCentroid in the classification of biomedical omics data, particularly in the field of precision medicine.
## Environmental configuration
nltk 0.9.1, 
numpy 2.2.0, 
pandas 1.4.4, 
partd 0.26.2, 
python 4.2.1, 
scikit-learn 1.4.0, 
scipy 20.3.0, 
tqdm 1.5.6, 
## Dataset
- The experimental data of DeepCentroid are divided into three aspects: early cancer diagnosis aspects, cancer prognosis aspects and drug sensitivity prediction.  
- The early cancer diagnosis data are plasma cell-free DNA whole genome sequencing data from lung cancer samples and controls.  
- Cancer prognostic data were obtained from Gene Expression Omnibus, the training data were GSE2034 and the independent validation data were GSE7390 , GSE11121 and GSE12093. These four datasets include transcriptomic data from breast cancer patients, as well as clinical data such as follow-up information.  
- Drug sensitivity prediction data were obtained from Genomics of Drug Sensitivity in Cancer, including DNA methylation data in multiple cell lines , Robust Multi-Array Average Normalized gene expression data, and drug sensitivity data for cell lines . A more specific presentation of the data can be found in the supplemental content.  
[You can download our data here](https://dmpmlab.github.io/Packages.html)
## Model structure
![图片1](https://github.com/xiexiexiekuan/DeepCentroid/assets/49866501/148ac724-c149-40d3-bb5b-c564537c516f)
- The DeepCentroid package provides a binary computational model for precision medicine that can integrate heterogeneous data.  
- It is divided into two phases, feature scanning, and cascade learning, and has stable and efficient predictions in the fields of early diagnosis, prognosis estimation, and anti-cancer drug sensitivity of cancer. 
- In the feature scanning stage, DeepCentroid can process heterogeneous information of different dimensions and dynamically divide the feature set according to the data size. In the cascade learning stage, DeepCentroid adopts a deeply cascaded strategy and integrates centroid classifiers to make predictions.
## User Manual
1. Download the project source code and data locally. There are three types of experimental data, among which early diagnosis and prognosis data also have independent validation data. Please pay attention to distinguishing.
2. Configure Python dependencies according to environmental requirements.
3. You can use the provided interface for personalized modifications.
- Selection of Datasets
- The size and quantity of gene sets
- Introduction of known gene sets
- Key gene result statistics
- Output level of model training
## Contact us
If you have any questions or needs regarding usage, you can edit the email to:1192450429@qq.com









