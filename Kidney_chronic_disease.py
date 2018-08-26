
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import Imputer,StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
import statsmodels.formula.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def backwardElimination(x,Y,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x

#Dataset loading
dataset = pd.read_csv(r"C:\Users\rg\Desktop\Guvi'sHackathon\ckdisease (2)\kidney_disease.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 25:26].values
Z = dataset.iloc[:, :-1]


#filling missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:, 1:25]) #DATA THAT HAS TO BE CLEANED
X[:, 1:25] = imputer.transform(X[:, 1:25])

#data for predicting anaemic
X_anaemic_check = X[:, 1:24]
X_anaemic = X[:, 24:25]
X_hypertension = X[:, 1:19]
Y_hypertension = X[:,19:20]

#Scaling Data (for both anaemia and CKD)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_anaemic = sc_X.fit_transform(X_anaemic)

#optimized feature data
X_opt = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

#feature screening(Greedy feaure selection (or) 
sl = 0.05
X_opt = backwardElimination(X_opt,Y,sl)

#MachineLearning part
CKD = DecisionTreeClassifier()
CKD.fit(X_opt,Y)

#Scaling and feature screening Anaemic,Hypertension data
X_anaemic_check = sc_X.fit_transform(X_anaemic_check)
X_anaemic_check = backwardElimination(X_anaemic_check,X_anaemic,sl)
X_hypertension = sc_X.fit_transform(X_hypertension)
X_hypertension_opt = backwardElimination(X_hypertension,Y_hypertension,sl)

#MachineLearning for anaemic,hypertension
anaemic = DecisionTreeClassifier()
X_anaemic_check = X_anaemic_check.astype(int)
X_anaemic = X_anaemic.astype(int)
anaemic.fit(X_anaemic_check,X_anaemic)
X_hypertension = X_hypertension.astype(int)
Y_hypertension = Y_hypertension.astype(int)
predict_hypertension = DecisionTreeClassifier()
predict_hypertension.fit(X_hypertension_opt,Y_hypertension)

#Input Dataset 
loc = input("Enter Location : ")

#loading dataset
dataset1 = pd.read_csv(loc)
final_x = dataset1.iloc[:, :-1].values

#Data preprocessing
imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0)
imputer = imputer.fit(final_x[:, 1:25]) #DATA THAT HAS TO BE CLEANED
final_x[:, 1:25] = imputer.transform(final_x[:, 1:25])
final_x_anaemic_check = X[:, 1:24]
final_x_hypertension = X[:, 1:19]

#Scaling Data
final_sc_x = StandardScaler()
final_x = final_sc_x.fit_transform(final_x)
final_x_anaemic_check = final_sc_x.fit_transform(final_x_anaemic_check)
final_x_hypertension = final_sc_x.fit_transform(final_x_hypertension)

#optimized final dataset thru feature scaling
final_x_opt = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
final_x_opt = backwardElimination(final_x,Y,sl)
final_x_anaemic_check = backwardElimination(final_x_anaemic_check,X_anaemic,sl)
final_x_hypertension_opt = backwardElimination(final_x_hypertension,Y_hypertension,sl)

#predictions of CKD and Anaemia
final_ckd_predictions = CKD.predict(final_x_opt)
final_anaemic_predictions = anaemic.predict(final_x_anaemic_check)
final_hypertension_predictions = predict_hypertension.predict(final_x_hypertension_opt)


# ###### 

# In[ ]:


C:\Users\rg\Desktop\Guvi'sHackathon\ckdisease (2)\kidney_disease.csv

