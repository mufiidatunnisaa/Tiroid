import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import csv
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import math

#%%imput data
data_tiroid = np.genfromtxt("data_tiroid_missing.csv", delimiter=',')
#%%imputasi
missing_df = pd.DataFrame(data_tiroid)
all_df = []
for label, label_missing_df in missing_df.groupby(5):
    all_df.append(label_missing_df.fillna(label_missing_df.mean().round()))
all_df = pd.concat(all_df)
print(all_df)


