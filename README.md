# IoTDevID: A Behavior-Based Device Identification Method for the IoT

# Overview
In this repository you will find a Python implementation of the methods in the paper IoTDevID: A Behavior-Based Device Identification Method for the IoT.

[*Kahraman  Kostas,  Mike  Just,  and  Michael  A.  Lones. IoTDevID: A Behavior-Based Device Identification Method for the IoT, IEEE Internet of Things Journal, 2022.*](https://ieeexplore.ieee.org/document/9832419)


# What is IoTDevID?


Device identification is one way to secure a network of IoT devices, whereby devices identified as suspicious can subsequently be isolated from a network. In this study, we present a machine learning-based method, *IoTDevID*, that recognises devices through characteristics of their network packets. As a result of using a rigorous feature analysis and selection process, our study offers a generalizable and realistic approach to modelling device behavior, achieving high predictive accuracy across two public datasets. The model's underlying feature set is shown to be more predictive than existing feature sets used for device identification, and is shown to generalise to data unseen during the feature selection process. Unlike most existing approaches to IoT device identification, IoTDevID is able to detect devices using non-IP and low-energy protocols.


# Requirements and Infrastructure: 

Wireshark and Python 3.6 were used to create the application files. Before running the files, it must be ensured that [Wireshark](https://www.wireshark.org/), [Python 3.6+](https://www.python.org/downloads/) and the following libraries are installed.

| Library | Task |
| ------ | ------ |
|[ Scapy ](https://scapy.net/)| Packet(Pcap) crafting |
|[ tshark ](https://scapy.net/)| Packet(Pcap) crafting |
|[ Sklearn ](http://scikit-learn.org/stable/install.html)| Machine Learning & Data Preparation |
|[ xverse ](https://pypi.org/project/xverse/)| Feature importance/voting |
| [ Numpy ](http://www.numpy.org/) |Mathematical Operations|
| [ Pandas  ](https://pandas.pydata.org/pandas-docs/stable/install.html)|  Data Analysis|
| [ Matplotlib ](https://matplotlib.org/users/installing.html) |Graphics and Visuality|
| [Seaborn ](https://seaborn.pydata.org/) |Graphics and Visuality|
| [graphviz ](https://graphviz.readthedocs.io/en/stable/manual.html) |Graphics and Visuality|



The technical specifications of the computer used for experiments are given below.

|  | |   |
| ------ |--|  ------ |
|Central Processing Unit|:|Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz 2.90 GHz|
| Random Access Memory	|:|	8 GB (7.74 GB usable)|
| Operating System	|:|	Windows 10 Pro 64-bit |
| Graphics Processing Unit	|:|	AMD Readon (TM) 530|

# Implementation: 

The implementation phase consists of 5 steps, which are:

* Feature Extraction
* Feature Selection 
* Algorithm Selection 
* Performance Evaluation
* Comparison with Previous Work


Each of these steps is implemented using one or more Python files. The same file was saved with both "py" and "ipynb" extensions. The code they contain is exactly the same. The file with the ipynb extension has the advantage of saving the state of the last run of that file and the screen output. Thus, screen output can be seen without re-running the files. Files with the ipynb extension can be run using [jupyter notebook](http://jupyter.org/install). 









## 01 Feature Extraction (PCAP2CSV) 
#### Section III.C in the article

There are four files relevant to this section:

* [01.1 Aalto feature extraction IoTDevID](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0001%20Feature%20Extraction%20-%20PCAP2CSV/01.1%20Aalto%20feature%20extraction%20IoTDevID.ipynb)
* [01.2 Aalto feature extraction IoTSense - IoT Sentinel](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0001%20Feature%20Extraction%20-%20PCAP2CSV/01.2%20Aalto%20feature%20extraction%20IoTSense-%20IoT%20Sentinel.ipynb)
* [01.3 UNSW feature extraction IoTDevID](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0001%20Feature%20Extraction%20-%20PCAP2CSV/01.3%20UNSW%20feature%20extraction%20IoTDevID.ipynb)
* [01.4 UNSW feature extraction IoTSense - IoT Sentinel](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0001%20Feature%20Extraction%20-%20PCAP2CSV/01.4%20UNSW%20feature%20extraction%20IoTSense-%20IoT%20Sentinel.ipynb)


These files convert the files with pcap extension to single packet-based, CSV extension fingerprint files (IoT Sentinel, IoTSense, IoTDevID individual packet based feature sets) and creates the labeling.


The processed datasets are shared in the repository. However, raw versions of the datasets used in the study and their addresses are given below.

| Dataset | capture year | Number of Devices | Type |
|---|---|---|---|
|[ Aalto University ](https://research.aalto.fi/en/datasets/iot-devices-captures)| 2016|31|Benign|
|[ UNSW-Sydney IEEE TMC ](https://iotanalytics.unsw.edu.au/iottraces)| 2016|31|Benign|
|[ UNSW-Sydney ACM SOSR](https://iotanalytics.unsw.edu.au/attack-data)| 2018|28|Benign & Malicious|




Since the UNSW data are very large, we filter the data on a device and session basis. You can access the Pcap files obtained from this filtering process from [ this link (Used Pcap Files)](https://drive.google.com/file/d/1RSnQJNTHj8FoS1KvBxbCGaS4sYmX3sPF/view).


In addition, the CSVs.zip file contains the feature sets that are the output of this step and that we used in our experiments. These files:
* Aalto_test_IoTDevID.csv
* Aalto_train_IoTDevID.csv
* Aalto_IoTSense_Test.csv
* Aalto_IoTSense_Train.csv
* Aalto_IoTSentinel_Test.csv
* Aalto_IoTSentinel_Train.csv
* UNSW_test_IoTDevID.csv
* UNSW_train_IoTDevID.csv
* UNSW_IoTSense_Test.csv
* UNSW_IoTSense_Train.csv
* UNSW_IoTSentinel_Test.csv
* UNSW_IoTSentinel_Train.csv




## 02 Feature Selection 

#### Section IV.A in the article
There are three files relevant to this section.
   
* **[02.1 Feature importance voting and pre-assessment of features: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0002%20Feature%20Selection/02.1%20Feature%20importance%20voting%20and%20pre-assessment%20of%20features.ipynb)**
This file calculates the importance scores for each feature using six feature score calculation methods. It then votes for features using these scores. It lists the feature scores and the votes they have received and shows them on a plot. The six feature importance score calculation methods used are as follows.

    * Information Value using Weight of evidence.
    * Variable Importance using Random Forest.
    * Recursive Feature Elimination.
    * Variable Importance using Extra trees classifier.
    * Chi-Square best variables.
    *  L1-based feature selection.

* **[02.2 Comparison of isolated data and CV methods: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0002%20Feature%20Selection/02.2%20Comparison%20of%20isolated%20and%20CV%20methods.ipynb)**
In this file, the results of the isolated test-training data and the cross-validated data are compared.


* **[02.3 Feature selection process using genetic algorithm: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0002%20Feature%20Selection/02.3%20Feature%20selection%20process%20using%20genetic%20algorithm%20Aalto.ipynb)**
In this file, feature selection is performed by using a genetic algorithm.

## 03 Algorithm Selection 
#### Section IV.B in the article

There are two files relevant to this section.


* **[03.1 Hyperparameter Optimization: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0003%20Algorithm%20Selection/03.%201%20Hyperparameter%20Optimization.ipynb)**
In this file, hyperparameter optimization is applied via [sklearn-Randomizedsearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to the machine learning models being used. These machine learning models are:
    * Decision Trees (DT)
    * Naïve Bayes (NB)
    * Gradient Boosting (GB)
    * k-Nearest Neighbours (kNN)
    * Random Forest (RF)
    * Support Vector Machine (SVM)

* **[03. 2 Classification of Individual packets for Aalto Dataset: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0003%20Algorithm%20Selection/03.%202%20Classification%20of%20Individual%20packets%20for%20Aalto%20Dataset.ipynb)**
This file trains machine learning models using the individual packets of Aalto University dataset using the methods mentioned above and the optimised hyperparameters.



## 04 Performance Evaluation
#### Section V  in the article
There are four files relevant to this section. In our experiments above, we found that DT offers the best balance between predictive performance and inference time among other machine learning methods. Therefore, only DT is used in all our subsequent experiments.

* **[04.1 Determination of aagregetion size: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0004%20Performance%20Evaluation/04.1%20Determination%20of%20aagregetion%20size.ipynb)**
In this file, different aggregation sizes are tested. For this purpose, groups of different sizes  (from 2 to 25) are formed and the performance results of these groups are observed.

* **[04.2 Classification of ind-aag-mixed packets for Aalto Dataset: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0004%20Performance%20Evaluation/04.2%20Classification%20of%20ind-aag-mixed%20packets%20for%20Aalto%20Dataset.ipynb)**
In this file, results are obtained for the Aalto dataset using individual, aggregated and mixed methods. A group size of 13 was used in the aggregation  operations.


* **[04.3 Classification of  ind-aag-mixed packets for UNSW Dataset: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0004%20Performance%20Evaluation/04.3%20Classification%20of%20%20ind-aag-mixed%20packets%20for%20UNSW%20Dataset.ipynb)**
In this file, results are obtained for the UNSW dataset using individual, aggregated and mixed methods. A group size of 13 was used in the aggregation  operations.


* **[04.4 Aalto results with combined labels: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0004%20Performance%20Evaluation/04.4%20Aalto%20results%20with%20combined%20labels.ipynb)**
In this file,  to deal with lower performance caused by the fact that the Aalto dataset contains many very similar devices, these similar devices are considered as a group and collected under the same label. 


## 05 Comparison with Previous Work
#### Section VI  in the article

There are two files relevant to this section.

* **[05.1 Aalto IoTSense & IoTSentinel  Normal, Aagregeted, Mixed Results: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0005%20compare%20with%20others/05.1%20Aalto%20IoTSense%20%26%20IoTSentinel%20%20Normal%2C%20Aagregeted%2C%20Mixed%20Results.ipynb)**
This file trains machine learning models using Aalto University data for 3 studies (IoTDevID, [IoTSense](https://dl.acm.org/doi/pdf/10.1145/3266444.3266452), [IoT Sentinel](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7980167)) with an individual, aggregated and mixed approach in order to compare the feature set  performances.

* **[05.2 UNSW IoTSense & IoTSentinel  Normal, Aagregeted, Mixed Results: ](https://github.com/kahramankostas/IoTDevIDv2/blob/main/0005%20compare%20with%20others/05.2%20UNSW%20IoTSense%20%26%20IoTSentinel%20%20Normal%2C%20Aagregeted%2C%20Mixed%20Results.ipynb)**
This file trains machine learning models using UNSW data for 3 studies (IoTDevID, [IoTSense](https://dl.acm.org/doi/pdf/10.1145/3266444.3266452), [IoT Sentinel](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7980167)) with an individual, aggregated and mixed approach in order to compare the feature set performances.


# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


# Citations
If you use the source code please cite the following paper:

```
@ARTICLE{9832419,
  author={Kostas, Kahraman and Just, Mike and Lones, Michael A.},
  journal={IEEE Internet of Things Journal}, 
  title={IoTDevID: A Behavior-Based Device Identification Method for the IoT}, 
  year={2022},
  volume={9},
  number={23},
  pages={23741-23749},
  doi={10.1109/JIOT.2022.3191951}}
}
```

Contact:
*Kahraman Kostas
kahramankostas@gmail.com*


