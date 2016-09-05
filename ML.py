# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#shape, this will allow us to get how many instances (rows) and how many attributes(columns)
#the data contains with the shape property
print(dataset.shape)

#Looking at the dataset itself (first 20 rows)
print(dataset.head(20))

#Statistical summary 
print(dataset.describe())

#class distribution
print(dataset.groupby('class').size())

#Data Visualization

#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms 
dataset.hist()
plt.show()

#scatter plot matrix
scatter_matrix(dataset)
plt.show()

#split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Test options and evaluation metrix
num_folds = 10
num_instances = len(X_train)
seed = 7

scoring = 'accuracy' #This is being used to evaluate models

#Evaluating 6 different Algorithims

models = [] 
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds = num_folds, random_state = seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#Output
#LR: 0.966667 (0.040825)
#LDA: 0.975000 (0.038188)
#KNN: 0.983333 (0.033333)
#CART: 0.975000 (0.038188)
#NB: 0.975000 (0.053359)
#SVM: 0.991667 (0.025000)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#With the current dataset, the KNN model was the most accurate

#Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


