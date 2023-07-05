

#Importing Library
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


#import dataset


dataset = dataset.fillna()

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Take care of missing data  
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

imputer = imputer.fit(X[:,1:3]) 
X[:,1:3] = imputer.transform(X[:,1:3])


from sklearn.preprocessing import LabelEncoder 
labelencoder_x = LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])

np.unique(X[:,0])

