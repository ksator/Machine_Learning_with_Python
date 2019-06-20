Please visit the [wiki](https://github.com/ksator/Machine_Learning_with_Python/wiki)  


# Documentation structure

- [manipulate dataset with pandas](#manipulate-dataset-with-pandas)  
- [Remove irrelevant features to reduce overfitting](#remove-irrelevant-features-to-reduce-overfitting)  
  - [Recursive Feature Elimination](#recursive-feature-elimination)  



# Remove irrelevant features to reduce overfitting 

To prevent overfitting, improve the data by removing irrelevant features. 

## Recursive Feature Elimination

The class `RFE` (Recursive Feature Elimination) from the `feature selection` module from the python library scikit-learn recursively removes features. It selects features by recursively considering smaller and smaller sets of features. It first trains the classifier on the initial set of features. It trains a classifier multiple times using smaller and smaller features set. After each training, the importance of the features is calculated and the least important feature is eliminated from current set of features. That procedure is recursively repeated until the desired number of features to select is eventually reached. RFE is able to find out the combination of features that contribute to the prediction. You just need to import RFE from sklearn.feature_selection and indicate which classifier model to use and the number of features to select.  

Here's how you can use the class `RFE` in order to find out the combination of important features.  

We will use this basic example [recursive_feature_elimination.py](recursive_feature_elimination.py)  

Load LinearSVC class from Scikit Learn library  
LinearSVC performs classification. LinearSVC is similar to SVC with parameter kernel='linear'. LinearSVC finds the linear separator that maximizes the distance between itself and the closest/nearest data point point  
```
>>> from sklearn.svm import LinearSVC
```
load RFE (Recursive Feature Elimination). RFE is used to remove features  
```
>>> from sklearn.feature_selection import RFE
```
load the iris dataset  
```
from sklearn import datasets
dataset = datasets.load_iris()
```
the dataset has 150 items, each item has 4 features (sepal length, sepal width, petal length, petal width)
```
>>> dataset.data.shape
(150, 4)
>>> dataset.feature_names
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```
instanciate the LinearSVC class  
```
>>> svm = LinearSVC(max_iter=5000)
```
instanciate the RFE class. select the number of features to keep (3 in that example). select the classifier model to use
```
>>> rfe = RFE(svm, 3)
```
use the iris dataset and fit
```
rfe = rfe.fit(dataset.data, dataset.target)
```
print summaries for the selection of attributes
```
>>> print(rfe.support_)
[False  True  True  True]
>>> print(rfe.ranking_)
[2 1 1 1]
```
So, sepal length is not selected. The 3 selected features are sepal width, petal length, petal width.  
