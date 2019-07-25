import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import csv
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import math

#%%inputasi
data_tiroid = pd.read_csv("data_tiroid_missing.csv", header=None)
data_tiroid = data_tiroid.replace('?', np.nan)
label = data_tiroid.iloc[:,5:6]
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(data_tiroid)
SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='mean', verbose=0)
data_tiroid = impute.transform(data_tiroid).tolist()

with open('tiroid.csv', 'w') as csvFile : #saving into csv
    writer = csv.writer(csvFile)
    writer.writerows(data_tiroid)
    csvFile.close()

#%%errorasli
df = pd.read_csv("tiroid.csv", header=None)
jumlahData = len(df)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(df, label.values)
result = knn.predict(df)
diff = 0 #mencari differences
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff+=1
error = (diff/jumlahData)*100

#%%minamx
scaler = MinMaxScaler()
scaler.fit(data_tiroid)
MinMaxScaler(copy=True, feature_range=(0, 1))
data_minmax = scaler.transform(data_tiroid).tolist()
with open('minmax_tiroid.csv', 'w') as csvFile : #save minmax result into csv
    writer = csv.writer(csvFile)
    writer.writerows(data_minmax)
    csvFile.close()

diff_minmax = 0 #count differences minmax->to count error
knn.fit(data_minmax, label) #input data training ke fungsi knn
result = knn.predict(data_minmax) #melakukan prediksi knn
for i, item in enumerate(label):
    if(item!=result[i]) : #jika beda dengan result->eror++
        diff_minmax+=1
error_minmax = (diff_minmax/jumlahData)*100

#%%zscore
data_zscore = stats.zscore(data_tiroid).tolist() #perhitungan zscore
with open('zscore_tiroid.csv', 'w') as csvFile :
    writer = csv.writer(csvFile)
    writer.writerows(data_zscore)
    csvFile.close()

diff_zscore = 0 #perhitngan eror
knn.fit(data_zscore, label.values)
result = knn.predict(data_zscore)
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff_zscore+=1
error_zscore = (diff_zscore/jumlahData)*100

#%%sigmoidal
data = pd.read_csv("zscore_tiroid.csv", header=None)


data[0] = (1-(2.718281828 ** (-1 * (data[0]))))/(1+(2.718281828 ** (-1 * (data[0]))))
data[1] = (1-(2.718281828 ** (-1 * (data[0]))))/(1+(2.718281828 ** (-1 * (data[1]))))
data[2] = (1-(2.718281828 ** (-1 * (data[0]))))/(1+(2.718281828 ** (-1 * (data[2]))))
data[3] = (1-(2.718281828 ** (-1 * (data[0]))))/(1+(2.718281828 ** (-1 * (data[3]))))
data[4] = (1-(2.718281828 ** (-1 * (data[0]))))/(1+(2.718281828 ** (-1 * (data[4]))))

print(data)

df = pd.read_csv("tiroid.csv", header=None)
label = df.iloc[:,5:6]
diff_sigmoidal= 0
knn.fit(data, label.values)
result = knn.predict(data)
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff_sigmoidal+=1
error_sigmoidal = (diff_sigmoidal/jumlahData)*100

print("error sebelum Normalisasi : ", error, "%")
print("error Minmax : ", error_minmax, "%")
print("error Zscore : ", error_zscore, "%")
print("error sigmoidal : ", error_sigmoidal, "%")