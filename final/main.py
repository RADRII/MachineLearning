import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from readfiles import readFiles

plt.rcParams['figure.figsize'] = [12, 7]

# Check if the condensed data file exists
# If it does, use that data, if not reread all of the csv files and condense them per day
if not os.path.exists("condenseddata.csv"):
    readFiles()

# Read in condensed data
df = pd.read_csv("condenseddata.csv")
dates=df.iloc[:,0]
dates = pd.to_datetime(dates)
avgUsages=df.iloc[:,1]

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
# Remove year from date and only mark month and weekday
months = []
weekDays = []
for date in dates:
    months.append(date.month)
    weekDays.append(date.day_of_week)
months = pd.Series(months, index = None)
weekDays = pd.Series(weekDays, index = None)

# Split into only precovid data
precovidBA = dates < "2020-03-15"
pcUsages = avgUsages[precovidBA]
pcMonths = months[precovidBA]
pcWeekDays = weekDays[precovidBA]

# X's: month, weekday
# Y: Usage
X = np.column_stack((pcWeekDays,pcMonths))

## Train and test various models to find best fit parameters
# Train
fig, ax = plt.subplots(figsize=(12, 7))

range = pd.date_range(start="2018-08-01",end="2020-03-15")
monthsTest = []
weekDaysTest = []
for date in range:
    monthsTest.append(date.month)
    weekDaysTest.append(date.day_of_week)
Xtest = np.column_stack((weekDaysTest,monthsTest))


degreePoly = [1, 2, 5, 7]
for degree in degreePoly:
    plt.plot(dates[precovidBA][::3], pcUsages[::3])
    polyX = PolynomialFeatures(degree).fit_transform(np.array(X))

    model = LinearRegression()
    model.fit(polyX, pcUsages)

    polyXTest = PolynomialFeatures(degree).fit_transform(np.array(Xtest))
    ypred = model.predict(polyXTest)

    plt.plot(range[::3],ypred[::3], alpha=0.6) #Plot every 3rd day to be able to read graph

    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Pre-Covid Linear Regression Model Predictions")
    stringdegree = "Degree= " + int.__str__(degree)
    plt.legend(["Real Data", stringdegree], loc='upper right',fancybox=True, shadow=True)
    plt.title("Compiled Bike Data")
    stringname = "featuremodel" + int.__str__(degree)
    plt.savefig('./Report/' + stringname)
    plt.show()






