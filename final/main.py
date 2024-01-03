import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from readfiles import readFiles

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
# Split into only precovid data
precovidBA = dates < "2020-03-15"
pcUsages = avgUsages[precovidBA]
pcdates = dates[precovidBA]


# X's: month, weekday
# Y: Usage



