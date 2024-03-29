import datetime
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from readfiles import readFiles

plt.rcParams['figure.figsize'] = [12, 7]

# Check if the condensed data file exists
# If it does, use that data, if not reread all of the csv files and condense them per day

# WARNING this process can take up to two hours!!
# The file already exists, if you want to test the data preposessing you can delete it but expect to be waiting a long time. 
if not os.path.exists("condenseddata.csv"):
    readFiles()

# Read in condensed data
df = pd.read_csv("condenseddata.csv")
dates=df.iloc[:,0]
dates = pd.to_datetime(dates)
avgUsages=df.iloc[:,1]

# Remove duplicates, theres just a few as some dates are in two files. They all have the same avgusage for both so its fine to just delete one.
duplicateBA = dates.duplicated(keep="first")
dates = dates[~duplicateBA].reset_index(drop=True)
avgUsages = avgUsages[~duplicateBA].reset_index(drop=True)

## Create and show graphs of the data
fig, ax = plt.subplots(figsize=(12, 7))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fmt = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_formatter(fmt)
plt.plot(dates[::3],avgUsages[::3]) #Plot every 3rd day to be able to read graph

plt.xticks(rotation=30)
plt.xlabel("Date")
plt.ylabel("Average Bikes Taken from All Stations")
plt.legend(["Usage vs Date"], loc='upper right',fancybox=True, shadow=True)
plt.title("Compiled Bike Data")

plt.savefig('./Report/CompiledData.png')
plt.show()

## Reorganize data for training
months = []
weekDays = []
years = []
days = []
for date in dates:
    months.append(date.month)
    weekDays.append(date.day_of_week)
    days.append(date.day)
    years.append(date.year)
months = pd.Series(months, index = None)
weekDays = pd.Series(weekDays, index = None)
days = pd.Series(days, index = None)
years = pd.Series(years, index = None)

# Split into only precovid data
precovidBA = dates < "2020-03-15"
pcDates = dates[precovidBA]
pcUsages = avgUsages[precovidBA]
pcMonths = months[precovidBA]
pcWeekDays = weekDays[precovidBA]
pcDays = days[precovidBA]
pcYears = years[precovidBA]

# X's: month, weekday
# Y: Usage
pcX = np.column_stack((pcWeekDays,pcMonths))
pcXY = np.column_stack((pcWeekDays,pcMonths, pcYears))
pcXD = np.column_stack((pcDays, pcWeekDays,pcMonths))
pcXYD = np.column_stack((pcDays, pcWeekDays,pcMonths, pcYears))

rangedate = pd.date_range(start="2018-08-01",end="2020-03-15")
monthsTest = []
weekDaysTest = []
daysTest = []
yearsTest = []
for date in rangedate:
    monthsTest.append(date.month)
    weekDaysTest.append(date.day_of_week)
    daysTest.append(date.day)
    yearsTest.append(date.year)
Xtest = np.column_stack((weekDaysTest,monthsTest))
XtestY = np.column_stack((weekDaysTest,monthsTest, yearsTest))
XtestD = np.column_stack((daysTest, weekDaysTest,monthsTest))
XtestYD = np.column_stack((daysTest, weekDaysTest,monthsTest, yearsTest))

#Remove dates that arent in training for scoring
dateisInTestingBA = rangedate.isin(pcDates)

#Train and test various models to find best variables
pcXs = [pcX, pcXY, pcXD, pcXYD]
Xtests = [Xtest, XtestY, XtestD, XtestYD]
names = [["Weekdays", "Months"], ["Weekdays", "Months", "Years"], ["Days", "Weekdays", "Months"], ["Days", "Weekdays", "Months", "Years"]]

for i in range(4):
    #Train and Test
    modelname = "Variables: "
    for j in range(len(names[i])):
        modelname = modelname + names[i][j] + " "
    plt.plot(pcDates[::3], pcUsages[::3])

    model = LinearRegression()
    model.fit(pcXs[i], pcUsages)

    ypred = model.predict(Xtests[i])

    #Dummy model comparison
    dr = DummyRegressor(strategy="mean")
    dr.fit(pcXs[i], pcUsages)
    ydummy = dr.predict(Xtests[i])

    #Plot
    plt.plot(rangedate[::3],ypred[::3], alpha=0.6) #Plot every 3rd day to be able to read graph
    plt.plot(rangedate[::3],ydummy[::3], alpha=0.6) #Plot every 3rd day to be able to read graph

    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Average Bikes Taken from All Stations")
    plt.legend(["Real Data", modelname, "Mean Dummy Model"], loc='upper right',fancybox=True, shadow=True)
    plt.title("Pre-Covid Linear Regression Model Predictions different Variables")
    stringname = "variablemodel" + int.__str__(i)
    plt.savefig('./Report/' + stringname)
    plt.show()

    #Score
    print("Linear model" + modelname + ":")
    print(r2_score(pcUsages, ypred[dateisInTestingBA]))
    print("Variable Coefficient")
    for j in range(len(model.coef_)):
        print("Coefficient theta -", names[i][j], ": ", model.coef_[j])

    print("Dummy model" + modelname + ":")
    print(r2_score(pcUsages, ydummy[dateisInTestingBA]))

## Train and test various models to find best degree parameter

degreePoly = [1, 2, 5, 7]
for degree in degreePoly:
    #Train and Test
    plt.plot(pcDates[::3], pcUsages[::3])
    polyX = PolynomialFeatures(degree).fit_transform(np.array(pcXY))

    model = LinearRegression()
    model.fit(polyX, pcUsages)

    polyXTest = PolynomialFeatures(degree).fit_transform(np.array(XtestY))
    ypred = model.predict(polyXTest)

    #Dummy model comparison
    dr = DummyRegressor(strategy="mean")
    dr.fit(polyX, pcUsages)
    ydummy = dr.predict(polyXTest)

    #Plot
    plt.plot(rangedate[::3],ypred[::3], alpha=0.6) #Plot every 3rd day to be able to read graph
    plt.plot(rangedate[::3],ydummy[::3], alpha=0.6) #Plot every 3rd day to be able to read graph

    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Average Bikes Taken from All Stations")
    stringdegree = "Degree= " + int.__str__(degree)
    plt.legend(["Real Data", stringdegree, "Mean Dummy Model"], loc='upper right',fancybox=True, shadow=True)
    plt.title("Pre-Covid Linear Regression Model Predictions")
    stringname = "featuremodel" + int.__str__(degree)
    plt.savefig('./Report/' + stringname)
    plt.show()

    #Score
    print("Linear model (Degree = ", degree, "):")
    print(r2_score(pcUsages, ypred[dateisInTestingBA]))

    print("Mean Dummy model (Degree = ", degree, "):")
    print(r2_score(pcUsages, ydummy[dateisInTestingBA]))

#Best model was the one with degree five
#Use model trained on pre-covid data to predict
    
#Train

polyX = PolynomialFeatures(5).fit_transform(np.array(pcX))
model = LinearRegression()
model.fit(polyX, pcUsages)

#Create predict X's
rangedate = pd.date_range(start="2018-08-01",end="2024-01-01")
monthsTest = []
weekDaysTest = []
for date in rangedate:
    monthsTest.append(date.month)
    weekDaysTest.append(date.day_of_week)
Xpredict = np.column_stack((weekDaysTest,monthsTest))
polyXp = PolynomialFeatures(5).fit_transform(np.array(Xpredict))

ypred = model.predict(polyXp)

#Plot
plt.plot(dates[::3],avgUsages[::3], alpha=0.6) #Plot every 3rd day to be able to read graph
plt.plot(rangedate[::3],ypred[::3], alpha=0.6) #Plot every 3rd day to be able to read graph
plt.axvline(x = datetime.date(2020, 3, 15), color = 'black')
plt.axvline(x = datetime.date(2022, 1, 22), color = 'black')

plt.xticks(rotation=30)
plt.xlabel("Date")
plt.ylabel("Average Bikes Taken from All Stations")
stringdegree = "Degree= " + int.__str__(5)
plt.legend(["Real Data", stringdegree, "Covid Start", "Covid Restrictions End"], loc='lower left',fancybox=True, shadow=True)
plt.title("Actual Covid Usages vs Model Predicted Usages ")
stringname = "covidDif" + int.__str__(5)
plt.savefig('./Report/' + stringname)
plt.show()

#Score
#Remove dates that arent in training for scoring
dateisInTestingBA1 = rangedate.isin(dates)

print("Linear model r2 score")
print(r2_score(avgUsages, ypred[dateisInTestingBA1]))
