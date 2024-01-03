# id:19-19-19-0 
# id:19--38-19-0 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

##Function to do a-e for each dataset
def partsatoe(filename):

    if filename=="week4a.csv":
        optimalk= 5
        optimaldegree= 5
    else:
        optimalk= 9
        optimaldegree= 2
        
    #Part A
    ##Read in data
    df = pd.read_csv(filename)
    X1=df.iloc[:,0]
    X2=df.iloc[:,1]
    X=np.column_stack((X1,X2))
    y=df.iloc[:,2]

    ##Graph data
    plt.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
    plt.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(["yReal","yPred"], loc='upper center',fancybox=True, shadow=True)
    plt.title("Raw Data")
    plt.show()

    ##Cross Validation for polynomial and C values
    plt.title("Logistic Regression Classifiers with Different Poly Degrees and Cis")

    polyRange = [1, 2, 5, 10]
    cRange = [0.01, 0.1, 1, 2]
    color = ['b', 'orange', 'g', 'r']
    counter = 0
    for i in polyRange:
        mean_f1=[]; std_f2=[]
        for c in cRange:
            model = LogisticRegression(C=c)
            temp=[]
            kf = KFold(n_splits=5)
            polyX = PolynomialFeatures(i).fit_transform(np.array(X))
            for train, test in kf.split(polyX):
                model.fit(polyX[train], y[train])
                ypred = model.predict(polyX[test])
                temp.append(f1_score(y[test], ypred))
            mean_f1.append(np.array(temp).mean())
            std_f2.append(np.array(temp).std())
        ##Graph
        markers, caps, bars = plt.errorbar(cRange, mean_f1, barsabove=True, yerr=std_f2, ecolor=color[counter], capsize=2, elinewidth=2, fmt="-o")
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
        counter = counter + 1

    plt.legend(["Degree: 1", "Degree: 2", "Degree: 5", "Degree: 10"], loc='lower right',fancybox=True, shadow=True)
    plt.xlabel("Ci")
    plt.ylabel("Mean F1")
    plt.show()

    ##Train optimal model
    polyX = PolynomialFeatures(optimaldegree).fit_transform(np.array(X))
    X_train, X_test, y_train, y_test = train_test_split(polyX, y)
    fig, (ax1, ax2) = plt.subplots(1, 2) 

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    ypred = logreg.predict(X_test)

    ##Train dummy
    dclf = DummyClassifier(strategy = 'uniform')
    dclf.fit(X_train, y_train)
    ydummy = dclf.predict(X_test)

    ##Graph
    ax1.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
    ax1.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
    ax1.scatter(X_test[ypred == 1,1], X_test[ypred == 1,2], color='r', marker='x')
    ax1.scatter(X_test[ypred == -1,1], X_test[ypred == -1,2], color='b', marker='x')

    ax1.set(xlabel='X1', ylabel='X2')
    ax1.set_title("Logistic Regression Classifier")

    ax2.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
    ax2.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
    ax2.scatter(X_test[ydummy == 1,1], X_test[ydummy == 1,2], color='r', marker='x')
    ax2.scatter(X_test[ydummy == -1,1], X_test[ydummy == -1,2], color='b', marker='x')

    ax2.set(xlabel='X1', ylabel='X2')
    ax2.set_title("Dummy (Random) Classifier")

    if filename=="week4a.csv":
        fig.suptitle("Logistic and Dummy Classifiers: Degree=5, C=1")
    else:
        fig.suptitle("Logistic and Dummy Classifiers: Degree=2, C=1")
    fig.legend(["y = 1", "y = -1", "yPred = 1", "yPred = -1"], ncol=4, loc='lower center',fancybox=True, shadow=True)
    plt.show()

    print("Logistic Regression F1 Score: ", f1_score(y_test, ypred))
    print("Dummy (Random) F1 Score: ", f1_score(y_test, ydummy))

    #Part B
    kRange = [2, 5, 9, 14]
    mean_f1=[]; std_f2=[]
    for k in kRange:
        model = KNeighborsClassifier(n_neighbors=k)
        temp=[]
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(f1_score(y[test], ypred))
        mean_f1.append(np.array(temp).mean())
        std_f2.append(np.array(temp).std())

    ##Graph
    markers, caps, bars = plt.errorbar(kRange, mean_f1, barsabove=True, yerr=std_f2, capsize=2, elinewidth=2, fmt="-o")
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]

    plt.legend(["F1 Score of Knn Models"], loc='lower right',fancybox=True, shadow=True)
    plt.title("F1 scores of Knn models with different k's")
    plt.xlabel("k")
    plt.ylabel("Mean F1")
    plt.show()

    ##Train optimal model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fig, (ax1, ax2) = plt.subplots(1, 2) 

    knn = KNeighborsClassifier(n_neighbors=optimalk)
    knn.fit(X_train, y_train)
    ypred = knn.predict(X_test)

    ##Train dummy
    dclf = DummyClassifier(strategy = 'uniform')
    dclf.fit(X_train, y_train)
    ydummy = dclf.predict(X_test)

    ##Graph
    ax1.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
    ax1.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
    ax1.scatter(X_test[ypred == 1,0], X_test[ypred == 1,1], color='r', marker='x')
    ax1.scatter(X_test[ypred == -1,0], X_test[ypred == -1,1], color='b', marker='x')

    ax1.set(xlabel='X1', ylabel='X2')
    ax1.set_title("Knn Classifier")

    ax2.scatter(X1[y == 1], X2[y == 1], color='pink', marker='o')
    ax2.scatter(X1[y == -1], X2[y == -1], color='paleturquoise', marker='o')
    ax2.scatter(X_test[ydummy == 1,0], X_test[ydummy == 1,1], color='r', marker='x')
    ax2.scatter(X_test[ydummy == -1,0], X_test[ydummy == -1,1], color='b', marker='x')

    ax2.set(xlabel='X1', ylabel='X2')
    ax2.set_title("Dummy (Random) Classifier")

    if filename=="week4a.csv":
        fig.suptitle("Knn and Dummy Classifiers: k=5")
    else:
         fig.suptitle("Knn and Dummy Classifiers: k=9")
    fig.legend(["y = 1", "y = -1", "yPred = 1", "yPred = -1"], ncol=4, loc='lower center',fancybox=True, shadow=True)
    plt.show()

    print("Knn F1 Score: ", f1_score(y_test, ypred))
    print("Dummy (Random) F1 Score: ", f1_score(y_test, ydummy))

    #Part C
    ##Using the models from the previous parts, test them all on the same y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    polyXtest = PolynomialFeatures(optimaldegree).fit_transform(np.array(X_test))
    polyXtrain = PolynomialFeatures(optimaldegree).fit_transform(np.array(X_train))

    logreg = LogisticRegression().fit(polyXtrain, y_train)
    knn = KNeighborsClassifier(n_neighbors=optimalk).fit(X_train, y_train)
    dclf = DummyClassifier(strategy='uniform').fit(X_train, y_train)
    
    predlog = logreg.predict(polyXtest)
    predknn = knn.predict(X_test)
    preddummy = dclf.predict(X_test)

    ##Print confusion matrices
    print("Confusion matrix for Logistic Regression Classifier")
    print(confusion_matrix(y_test, predlog))
    print("Confusion matrix for Knn Classifier")
    print(confusion_matrix(y_test, predknn))
    print("Confusion matrix for Dummy Classifier (random)")
    print(confusion_matrix(y_test, preddummy))

    #Part D

    ##Graph ROC curve for each model
    logreg = LogisticRegression().fit(polyXtrain, y_train)
    fpr, tpr, __ = roc_curve(y_test, logreg.predict_proba(polyXtest)[:, 1])
    plt.plot(fpr, tpr, color='orange', alpha=0.5, linewidth=1.8)

    knn = KNeighborsClassifier(n_neighbors=optimalk).fit(X_train, y_train)
    fpr, tpr, __ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, color='green', alpha=0.5, linewidth=1.8)

    dummy = DummyClassifier(strategy='uniform').fit(X_train, y_train)
    fpr, tpr, __ = roc_curve(y_test, dummy.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, color='blue', alpha=0.5, linewidth=1.8)

    plt.plot([0, 1], [0, 1], color='red', linestyle='dotted')

    ##Specify graph details
    plt.legend(['Logistic Regression', "Knn", "Dummy (Random)"], loc='lower right',fancybox=True, shadow=True)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC Curves for 3 models")
    plt.show()

    #Part E
    ##Graph f1 scores of each model for comparison
    classifiers = ["Logistic Regression", "Knn", "Dummy (Random)"]
    f1scores = [f1_score(y_test, logreg.predict(polyXtest)), f1_score(y_test, knn.predict(X_test)), f1_score(y_test, dummy.predict(X_test))]
    bar_colors = ['tab:orange', 'tab:green', 'tab:blue']

    for i in range(3):
        print(classifiers[i], " - ", f1scores[i])

    plt.bar(classifiers, f1scores, color=bar_colors)
    plt.title("F1 Scores of 3 different Classifiers")
    plt.show()

partsatoe("week4a.csv")
partsatoe("week4b.csv")
