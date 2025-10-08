import re
import csv
import pandas as pd


import datetime

timestamp = datetime.datetime.now().strftime('%H:%M:%S')

print(str(timestamp))


waypoints_lat = []
waypoints_lng = []

filename = 'Untitled map- demo_2.csv'

with open(filename, 'r',newline='') as csvfile:
    datareader = csv.reader(csvfile)
    #skips header row
    next(csvfile)
    for row in datareader:    
        numbers = re.findall(r"\d+", row[0])
        joined_lat = "".join(numbers[2] +'.'+ numbers[3])
        joined_lng = "".join('-'+numbers[0] +'.'+ numbers[1])
        waypoints_lat.append(eval(joined_lat))
        waypoints_lng.append(eval(joined_lng))

        
        
df = pd.DataFrame({'lat':waypoints_lat, 'long':waypoints_lng})
#df1 = df.drop_duplicates()
df.to_csv('cleaned_google_waypoints_demo_1.csv')
                
                    
