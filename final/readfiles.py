import pandas as pd
import os.path
from datetime import datetime

def readFiles():
    filenames = ["csv/yearly/2018-1.csv", "csv/yearly/2018-2.csv", "csv/yearly/2019-1.csv", "csv/yearly/2019-2.csv", "csv/yearly/2019-3.csv", "csv/yearly/2019-4.csv", "csv/yearly/2020-1.csv", "csv/yearly/2020-2.csv", "csv/yearly/2020-3.csv", "csv/yearly/2020-4.csv",
                 "csv/yearly/2021-1.csv", "csv/yearly/2021-2.csv", "csv/yearly/2021-3.csv", "csv/yearly/2021-4.csv", "csv/monthly/2022-1.csv", "csv/monthly/2022-2.csv", "csv/monthly/2022-3.csv", "csv/monthly/2022-4.csv", "csv/monthly/2022-5.csv", "csv/monthly/2022-6.csv", "csv/monthly/2022-7.csv", "csv/monthly/2022-8.csv", "csv/monthly/2022-9.csv", "csv/monthly/2022-10.csv", "csv/monthly/2022-11.csv", "csv/monthly/2022-12.csv",
                 "csv/monthly/2023-1.csv", "csv/monthly/2023-2.csv", "csv/monthly/2023-3.csv", "csv/monthly/2023-4.csv", "csv/monthly/2023-5.csv", "csv/monthly/2023-6.csv", "csv/monthly/2023-7.csv", "csv/monthly/2023-8.csv", "csv/monthly/2023-9.csv", "csv/monthly/2023-10.csv", "csv/monthly/2023-11.csv", "csv/monthly/2023-12.csv"]
    
    num = 1
    for filename in filenames:
        readFile(filename)
        print("Done ", num ," of 38")
        num = num + 1

def readFile(filename):
    df = pd.read_csv(filename)

    #Compress data to track each stations usage per date
    stations=df.iloc[:,0]
    times=df.iloc[:,1]
    bikes=df.iloc[:,6]

    compressed = []
    row = 0
    while row < len(times)-1:
       rowDate = datetime.strptime(times[row], '%Y-%m-%d %H:%M:%S').date()
       current = combinedStationTime(rowDate, stations[row], bikes[row])
       
       for c in compressed:
          if c.isDateEqual(current):
             c.addUsage(current)
             current = None
             break
        
       if current != None:
          compressed.append(current)

       print(row)
       row = row + 1

    #Write compressed data
    date = []
    avgusage = []
    for c in compressed:
       date.append(c.date)
       avgusage.append(c.getAvgUsage())

    toWrite = pd.DataFrame({'DATE': date, 'AVG USAGE':avgusage})
    if not os.path.exists("condenseddata.csv"):
        toWrite.to_csv('condenseddata.csv', mode='a', index=False)
    else:
       toWrite.to_csv('condenseddata.csv', mode='a', header=False, index=False)


class combinedStationTime:
  def __init__(self, date, station, bikes):
    self.date = date
    self.stations = [station]
    self.bikes = [bikes]
    self.usages = [0]

  def isDateEqual(self, other):
     if (self.date == other.date):
        return True
     return False
  
  def getAvgUsage(self):
     total = 0
     for usage in self.usages:
        total = total + usage
     return total / len(self.stations)

  
  def addUsage(self, other):
     if self.stations.count(other.stations[0]) > 0:
        index = self.stations.index(other.stations[0])
        if(other.bikes[0] < self.bikes[index]):
           self.usages[index] = self.usages[index] + (self.bikes[index] - other.bikes[0])
        self.bikes[index] = other.bikes[0]
     else:
       self.stations.append(other.stations[0])
       self.bikes.append(other.bikes[0])
       self.usages.append(0)


    

