from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load the data set
iris=load_iris()

X = iris.data
Y = iris.target

# split the data set in a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

# select a model and fit it
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# evaluate the trained model 
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)
print ("Accuracy of SVC fitted with iris data set:",accuracy_score(y_test,y_pred))
