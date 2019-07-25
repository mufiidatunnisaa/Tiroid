#%%zscore
import pandas as pd
import numpy as np
import csv
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

data_tiroid = pd.read_csv("tiroid.csv", header=None)
data_zscore = stats.zscore(data_tiroid).tolist()
label = data_tiroid.iloc[:,5:6]

with open('zscore_tiroid.csv', 'w') as csvFile :
    writer = csv.writer(csvFile)
    writer.writerows(data_zscore)
    csvFile.close()

jumlahData = len(data_tiroid)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_tiroid.values, label.values)
result = knn.predict(data_tiroid.values)
diff = 0
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff+=1
error = (diff/jumlahData)*100

knn.fit(data_zscore, label.values)
result = knn.predict(data_zscore)
diff_zscore = 0
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff_zscore+=1
error_zscore = (diff_zscore/jumlahData)*100

print("error sebelum Normalisasi : ", error, "%")
print("error Zscore : ", error_zscore, "%")