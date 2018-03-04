import numpy as np
import pandas as pd
from sklearn import model_selection, ensemble
import pickle

# load data set
df = pd.read_csv('train.csv')

# split into features and labels
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# split into test and training data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# specify classifier
clf = ensemble.RandomForestClassifier(n_estimators = 10)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy

# save model
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))