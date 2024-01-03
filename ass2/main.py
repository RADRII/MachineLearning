# id:11-11--11 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

##Function that adds the raw data to a graph
def addRawData(graph, name, X, Y):
    graph.scatter(X[:,0], X[:,1], Y, color='b', marker='o')
    graph.set_title(name)
    graph.set_xlabel("X1")
    graph.set_ylabel("X2")
    graph.set_zlabel("y")
    if name != "Lab (i)(a)":
        graph.plot([0],[0], linestyle="none", c='teal', alpha=0.3, marker = 'o')
        graph.legend(["yReal","yPred"], loc='upper center',fancybox=True, shadow=True)
    else:
        graph.legend(["yReal"], loc='upper center',fancybox=True, shadow=True)
        
    

#(i)(a)
##Read in data
df = pd.read_csv("week3.csv")
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

##Graph Data Points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(a)", X, y)
plt.show()

#(i)(b)
##Add polynomial features
polyX = PolynomialFeatures(5).fit_transform(np.array(X))

##Train Model
lasso1 = Lasso(alpha=0.5) #C = 1
lasso2 = Lasso(alpha=0.05) #C = 10
lasso3 = Lasso(alpha=0.005) #C = 100
lasso4 = Lasso(alpha=0.0005) #C = 1000
               
lasso1.fit(polyX, y)
lasso2.fit(polyX, y)
lasso3.fit(polyX, y)
lasso4.fit(polyX, y)

##Print Coefficients and Intercepts
print("Lasso C=1 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", lasso1.coef_[i])

print("Lasso C=10 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", lasso2.coef_[i])

print("Lasso C=100 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", lasso3.coef_[i])

print("Lasso C=1,000 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", lasso4.coef_[i])

print("Lasso C=1 Intercept")
print(lasso1.intercept_)
print("Lasso C=100 Intercept")
print(lasso2.intercept_)
print("Lasso C=1,000 Intercept")
print(lasso3.intercept_)
print("Lasso C=10,000 Intercept")
print(lasso4.intercept_)

#(i)(c)

xGrid = []
grid = np.linspace(-3,3)
for i in grid:
    for j in grid:
        xGrid.append([i,j])
xGrid = np.array(xGrid)
xGrid = PolynomialFeatures(5).fit_transform(xGrid)

##Predict
ypred1 = lasso1.predict(xGrid)
ypred2 = lasso2.predict(xGrid)
ypred3 = lasso3.predict(xGrid)
ypred4 = lasso4.predict(xGrid)

##Create Graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(B), C = 1", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred1, alpha=0.4, color='teal')
ax.set_zlim([-30, 30])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(B), C = 10", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred2, alpha=0.4, color='teal')
ax.set_zlim([-30, 30])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(B), C = 100", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred3, alpha=0.4, color='teal')
ax.set_zlim([-30, 30])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(B), C = 1,000", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred4, alpha=0.4, color='teal')
ax.set_zlim([-30, 30])
plt.show()

#(i)(e)
##Create and fit models
ridge1 = Ridge(alpha=0.5) #C = 1
ridge2 = Ridge(alpha=5.0) #C = 0.1
ridge3 = Ridge(alpha=50.0) #C = 0.01
ridge4 = Ridge(alpha=500.0) #C = 0.001

ridge1.fit(polyX, y)
ridge2.fit(polyX, y)
ridge3.fit(polyX, y)
ridge4.fit(polyX, y)

##Print Coefficients and Intercepts
print("Ridge C=1 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", ridge1.coef_[i])

print("Ridge C=0.1 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", ridge2.coef_[i])

print("Ridge C=0.01 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", ridge3.coef_[i])

print("Ridge C=0.001 Coefficients")
for i in range(21):
    print("Coefficient theta-", i+1, ": ", ridge4.coef_[i])

print("Ridge C=1 Intercept")
print(ridge1.intercept_)
print("Ridge C=0.1 Intercept")
print(ridge2.intercept_)
print("Ridge C=0.01 Intercept")
print(ridge3.intercept_)
print("Ridge C=0.001 Intercept")
print(ridge4.intercept_)

##Predict
ypred1 = ridge1.predict(xGrid)
ypred2 = ridge2.predict(xGrid)
ypred3 = ridge3.predict(xGrid)
ypred4 = ridge4.predict(xGrid)

##Create Graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(E), C = 1", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred1, alpha=0.4, color='teal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(E), C = 0.1", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred2, alpha=0.4, color='teal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(E), C = 0.01", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred3, alpha=0.4, color='teal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
addRawData(ax, "Lab (i)(E), C = 0.001", X, y)
ax.plot_trisurf(xGrid[:,1], xGrid[:,2], ypred4, alpha=0.4, color='teal')
plt.show()

#(ii)(a)

mean_error=[]; std_error=[]
Ci_range = [0.1, 1, 10, 100, 1000]
for Ci in Ci_range:
    model = Lasso(alpha=1/(2*Ci))
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(polyX):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

markers, caps, bars = plt.errorbar(Ci_range, mean_error, barsabove=True, yerr=std_error, ecolor='r', capsize=2, elinewidth=2, fmt="-o")
[bar.set_alpha(0.5) for bar in bars]
[cap.set_alpha(0.5) for cap in caps]

plt.xlabel('Ci')
plt.legend(["Mean squared error of different C's with error bars"], loc='upper center',fancybox=True, shadow=True)
plt.ylabel('Lasso Mean square error')
plt.xscale('log')
plt.show()

#(ii)(c)

mean_error=[]; std_error=[]
Ci_range = [0.001, 0.01, 0.1, 1, 10]
for Ci in Ci_range:
    model = Ridge(alpha=1/(2*Ci))
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(polyX):
        model.fit(polyX[train], y[train])
        ypred = model.predict(polyX[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

markers, caps, bars = plt.errorbar(Ci_range, mean_error, barsabove=True, yerr=std_error, ecolor='r', capsize=2, elinewidth=2, fmt="-o")
[bar.set_alpha(0.5) for bar in bars]
[cap.set_alpha(0.5) for cap in caps]

plt.xlabel('Ci')
plt.legend(["Mean squared error of different C's with error bars"], loc='upper center',fancybox=True, shadow=True)
plt.ylabel('Ridge Mean square error')
plt.xscale('log')
plt.show()