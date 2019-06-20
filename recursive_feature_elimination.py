# Load LinearSVC class from Scikit Learn library 
# LinearSVC is similar to SVC with parameter kernel='linear'
# LinearSVC performs classification
# LinearSVC finds the linear separator that maximizes the distance between itself and the closest/nearest data point point
from sklearn.svm import LinearSVC

# load RFE (Recursive Feature Elimination) 
# RFE is used to remove features
from sklearn.feature_selection import RFE

# load the iris dataset
from sklearn import datasets
dataset = datasets.load_iris()

# the dataset has 150 items, each item has 4 features (sepal length, sepal width, petal length, petal width) 
dataset.data.shape
print ("list of features in the dataset:")
print (dataset.feature_names)

# instanciate the LinearSVC class   
svm = LinearSVC(max_iter=5000)

# instanciate the RFE class 
# select the number of features to keep (3 in that example) 
# select the classifier model to use  
rfe = RFE(svm, 3)

# use the iris dataset and fit 
rfe = rfe.fit(dataset.data, dataset.target)

# print summaries for the selection of attributes
# sepal length is not selected. The 3 selected features are sepal width, petal length, petal width 
print ('selected features are:')
print(rfe.support_)
print(rfe.ranking_)

