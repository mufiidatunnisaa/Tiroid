import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import csv
from sklearn.neighbors import KNeighborsClassifier

data_tiroid = pd.read_csv("data_tiroid_missing.csv", header=None)
data_tiroid = data_tiroid.replace('?', np.nan)
label = data_tiroid.iloc[:,5:6]
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_tiroid)
SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='mean', verbose=0)
data_tiroid = impute.transform(data_tiroid).tolist()

with open('tiroid.csv', 'w') as csvFile :
    writer = csv.writer(csvFile)
    writer.writerows(data_tiroid)
    csvFile.close()

data_tiroid = pd.read_csv("data_tiroid_missing.csv", header=None)
data_tiroid = data_tiroid.replace('?', np.nan)
label = data_tiroid.iloc[:,5:6]
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_tiroid)
SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='mean', verbose=0)
data_tiroid = impute.transform(data_tiroid).tolist()

with open('tiroid.csv', 'w') as csvFile :
    writer = csv.writer(csvFile)
    writer.writerows(data_tiroid)
    csvFile.close()

df = pd.read_csv("tiroid.csv", header=None)
jumlahData = len(df)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(df, label.values)
result = knn.predict(df)
diff = 0
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff+=1
error = (diff/jumlahData)*100

#%%minamx
scaler = MinMaxScaler()
scaler.fit(data_tiroid)
MinMaxScaler(copy=True, feature_range=(0, 1))
data_minmax = scaler.transform(data_tiroid).tolist()
with open('minmax_tiroid.csv', 'w') as csvFile :
    writer = csv.writer(csvFile)
    writer.writerows(data_minmax)
    csvFile.close()

diff_minmax = 0
knn.fit(data_minmax, label)
result = knn.predict(data_minmax)
for i, item in enumerate(label):
    if(item!=result[i]) :
        diff_minmax+=1
error_minmax = (diff_minmax/jumlahData)*100

print("error sebelum Normalisasi : ", error, "%")
print("error Minmax : ", error_minmax, "%")