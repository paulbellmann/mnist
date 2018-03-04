from sklearn import datasets, svm, model_selection
import pickle

# load data set
digits = datasets.load_digits()

# split into features and labels
X = digits.data[:-10]
y = digits.target[:-10]

# split into test and training data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# specify classifier
clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy

# save model
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))