from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import preprocess

x, y = preprocess.preprocess("notMNIST_small")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

clf = svm.SVC()
clf.fit(x_train, y_train)
y_hat = clf.predict(x_test)

print (accuracy_score(y_test, y_hat))


