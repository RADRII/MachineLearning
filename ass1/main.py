# id:13-13--13 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#A(i)
##Read in data
df = pd.read_csv("week2.csv")
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

##Graph points
plt.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
plt.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')

##Specifiy Graph Details
plt.title("Lab 1 (A)")
plt.xlabel("X1")
plt.ylabel("X2")

#A(ii)

##Train Log Model
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
ypred = logreg.predict(X_test)

##Graph Predictions
plt.scatter(X_test[ypred == 1,0], X_test[ypred == 1,1], color='r', marker='x')
plt.scatter(X_test[ypred == -1,0], X_test[ypred == -1,1], color='b', marker='x')

#A(iii)

##Create and plot decision boundary
cf = logreg.coef_[0]
print("Coefficiants for X1 and X2 of the Logistic Regression model: ")
print(cf[0])
print(cf[1])
a = -cf[0] / cf[1]
xx = np.linspace(-1, 1)
yy = a * xx - (logreg.intercept_[0]) / cf[1]
plt.plot(xx, yy, 'k-')
print("Intercept for the Logistic Regression model: ")
print(logreg.intercept_[0])

##Add Legend and Plot
plt.legend(["y = 1", "y = -1", "yPred = 1", "yPred = -1"], ncol=4, loc='upper center',fancybox=True, shadow=True)
print("Simple Logictic Regressor Mean Squared Error: ")
print(mean_squared_error(y_test, ypred))
plt.show()


#B(i)
##Create Models, Train, and predict
svc1 = LinearSVC(C=0.001)
svc2 = LinearSVC(C=1)
svc3 = LinearSVC(C=50)
svc4 = LinearSVC(C=100)

svc1.fit(X_train, y_train)
svc2.fit(X_train, y_train)
svc3.fit(X_train, y_train)
svc4.fit(X_train, y_train)

ypred1 = svc1.predict(X_test)
ypred2 = svc2.predict(X_test)
ypred3 = svc3.predict(X_test)
ypred4 = svc4.predict(X_test)

#B(ii)
##Setup 4 graphs
fig, ax = plt.subplots(2, 2) 
fig.suptitle("Lab 1 (B)")
fig.tight_layout(pad=1.5)

##Graph data and add labels
ax[0, 0].scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax[0, 0].scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax[0, 0].set_title("C = 0.001") 
ax[0, 0].set(xlabel='X1', ylabel='X2')

ax[0, 1].scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax[0, 1].scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax[0, 1].set_title("C = 1") 
ax[0, 1].set(xlabel='X1', ylabel='X2')

ax[1, 0].scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax[1, 0].scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax[1, 0].set_title("C = 50") 
ax[1, 0].set(xlabel='X1', ylabel='X2')

ax[1, 1].scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax[1, 1].scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax[1, 1].set_title("C = 100") 
ax[1, 1].set(xlabel='X1', ylabel='X2')

##Add Predictions
ax[0, 0].scatter(X_test[ypred1 == 1,0], X_test[ypred1 == 1,1], color='r', marker='x')
ax[0, 0].scatter(X_test[ypred1 == -1,0], X_test[ypred1 == -1,1], color='b', marker='x')


ax[0, 1].scatter(X_test[ypred2 == 1,0], X_test[ypred2 == 1,1], color='r', marker='x')
ax[0, 1].scatter(X_test[ypred2 == -1,0], X_test[ypred2 == -1,1], color='b', marker='x')

ax[1, 0].scatter(X_test[ypred3 == 1,0], X_test[ypred3 == 1,1], color='r', marker='x')
ax[1, 0].scatter(X_test[ypred3 == -1,0], X_test[ypred3 == -1,1], color='b', marker='x')

ax[1, 1].scatter(X_test[ypred4 == 1,0], X_test[ypred4 == 1,1], color='r', marker='x')
ax[1, 1].scatter(X_test[ypred4 == -1,0], X_test[ypred4 == -1,1], color='b', marker='x')

##Add Decision Boundaries

cf = svc1.coef_[0]
a = -cf[0] / cf[1]
xx = np.linspace(-1, 1)
yy = a * xx - (svc1.intercept_[0]) / cf[1]
ax[0, 0].plot(xx, yy, 'k-')
print("Coefficiants for X1 and X2 of the C = 0.001: ")
print(cf[0])
print(cf[1])
print("Intercept for C = 0.001: ")
print(svc1.intercept_[0])

cf = svc2.coef_[0]
a = -cf[0] / cf[1]
xx = np.linspace(-1, 1)
yy = a * xx - (svc2.intercept_[0]) / cf[1]
ax[0, 1].plot(xx, yy, 'k-')
print("Coefficiants for X1 and X2 of the C = 1: ")
print(cf[0])
print(cf[1])
print("Intercept for C = 1: ")
print(svc2.intercept_[0])

cf = svc3.coef_[0]
a = -cf[0] / cf[1]
xx = np.linspace(-1, 1)
yy = a * xx - (svc3.intercept_[0]) / cf[1]
ax[1, 0].plot(xx, yy, 'k-')
print("Coefficiants for X1 and X2 of the C = 50: ")
print(cf[0])
print(cf[1])
print("Intercept for C = 50: ")
print(svc3.intercept_[0])

cf = svc4.coef_[0]
a = -cf[0] / cf[1]
xx = np.linspace(-1, 1)
yy = a * xx - (svc4.intercept_[0]) / cf[1]
ax[1, 1].plot(xx, yy, 'k-')
print("Coefficiants for X1 and X2 of the C = 100: ")
print(cf[0])
print(cf[1])
print("Intercept for C = 100: ")
print(svc4.intercept_[0])

fig.legend(["y = 1", "y = -1", "yPred = 1", "yPred = -1"], ncol=4, loc='lower center',fancybox=True, shadow=True)

print("SVC (C = 0.001) Mean Squared Error: ")
print(mean_squared_error(y_test, ypred1))
print("SVC (C = 1) Mean Squared Error: ")
print(mean_squared_error(y_test, ypred2))
print("SVC (C = 50) Mean Squared Error: ")
print(mean_squared_error(y_test, ypred3))
print("SVC (C = 100) Mean Squared Error: ")
print(mean_squared_error(y_test, ypred4))

plt.show() 

#C(i)
##Create Graphs
fig, (ax1, ax2) = plt.subplots(1, 2) 
fig.suptitle("Lab 1 (C)")
fig.tight_layout(pad=1.5)

##Graph points
ax1.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax1.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax1.set_title("Logictic Regression Model") 
ax1.set(xlabel='X1', ylabel='X2')

ax2.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
ax2.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
ax2.set_title("Dummy Model") 
ax2.set(xlabel='X1', ylabel='X2')

##Add squares of features to X's
X1X = [x ** 2 for x in X1]
X2X = [x ** 2 for x in X2]
Xs=np.column_stack((X1,X2,X1X,X2X))

##Train Log Model
X_train, X_test, y_train, y_test = train_test_split(Xs, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
ypred = logreg.predict(X_test)

#C(ii)
##Graph Predictions
ax1.scatter(X_test[ypred == 1,0], X_test[ypred == 1,1], color='r', marker='x')
ax1.scatter(X_test[ypred == -1,0], X_test[ypred == -1,1], color='b', marker='x')

#C(iii)
dclf = DummyClassifier(strategy = 'most_frequent')
dclf.fit(X_train, y_train)
ydummy = dclf.predict(X_test)

ax2.scatter(X_test[ydummy == 1,0], X_test[ydummy == 1,1], color='r', marker='x')
ax2.scatter(X_test[ydummy == -1,0], X_test[ydummy == -1,1], color='b', marker='x')

fig.legend(["y = 1", "y = -1", "yPred = 1", "yPred = -1"], ncol=4, loc='lower center',fancybox=True, shadow=True)

cf = logreg.coef_[0]
print("Coefficiants for 4 feature Logistic Regression: ")
print(cf[0])
print(cf[1])
print(cf[2])
print(cf[3])
print("Intercept: ")
print(logreg.intercept_[0])

print("Logistic Regressor Mean Squared Error: ")
print(mean_squared_error(y_test, ypred))
print("Dummy Model Mean Squared Error: ")
print(mean_squared_error(y_test, ydummy))

plt.show() 