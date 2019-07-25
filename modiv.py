import pandas as pd
import numpy as np
import imputation as imp
import matplotlib.pyplot as plt
import matplotlib

data = pd.read_csv("data_ruspini_missing.csv", header=None)
print(data)
# label = data_tiroid.iloc[:,3:4]

# for i, item in enumerate(label.values)
#     if(item=='1') :
#         label1 = np.array(item.float)
        

# data = imp.imputation('data_ruspini_missing.csv')

# data_real = pd.read_csv("tiroid.csv", header=None)
# data_real = np.array(data_real.float)
# data_real[:,2]=data_real[:,2]+4

# data_merge=np.concatenate((data,data_real),axis=0)
# a=data_merge[:,0]
# b=data_merge[:,1]
# c=np.array(data_merge[:,2],int)
# d=np.zeros(len(c),dtype=object)
# color=['blue','red','black','yellow','cyan','green','purple']
# for i in range(0,len(c)): d[i]=color[c[i]-1]
#     fig, ax = plt.subplots()
#     ax.scatter(a,b,c=d)