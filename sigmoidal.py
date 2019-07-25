#%%sigmoidal
import pandas as pd
import numpy as np
import csv
import math
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

data_tiroid = pd.read_csv("zscore_tiroid.csv", header=None)
# np.array(data_tiroid, dtype=float)


data_tiroid[0] = (1-(2.718281828 ** (-1 * (data_tiroid[0]))))/(1+(2.718281828 ** (-1 * (data_tiroid[0]))))
data_tiroid[1] = (1-(2.718281828 ** (-1 * (data_tiroid[0]))))/(1+(2.718281828 ** (-1 * (data_tiroid[1]))))
data_tiroid[2] = (1-(2.718281828 ** (-1 * (data_tiroid[0]))))/(1+(2.718281828 ** (-1 * (data_tiroid[2]))))
data_tiroid[3] = (1-(2.718281828 ** (-1 * (data_tiroid[0]))))/(1+(2.718281828 ** (-1 * (data_tiroid[3]))))
data_tiroid[4] = (1-(2.718281828 ** (-1 * (data_tiroid[0]))))/(1+(2.718281828 ** (-1 * (data_tiroid[4]))))

print(data_tiroid)

df = pd.read_csv("tiroid.csv", header=None)
label = df.iloc[:,5:6]

jumlahData = len(label)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_tiroid, label.values)
result = knn.predict(data_tiroid)
diff_sigmoidal= 0
for i, item in enumerate(label.values):
    if(item!=result[i]) :
        diff_sigmoidal+=1
error_sigmoidal = (diff_sigmoidal/jumlahData)*100
print("error sigmoidal : ", error_sigmoidal, "%")