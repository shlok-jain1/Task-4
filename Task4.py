#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
data = pd.read_csv("C:\\Users\\Admin\\Downloads\\Task 4 dataset\\Housing.csv")
print(data.head())
print(data.isnull().sum()) #there are no missing values
print("\nData distribution of Prices\n",data["price"].describe())
print("\nData distribution of Area\n",data["area"].describe())

def outliers(data,column):
    ob = []
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3-Q1
    lb = Q1 - 3*IQR
    ub = Q3 + 3*IQR
    ob = data[column][(data[column] < lb) | (data[column] > ub)]
    return(ob)
outliers(data,"price")
outliers(data,"area")

from sklearn.preprocessing import StandardScaler

def scale(data,col):
    scaler = StandardScaler()
    scaled = data.copy()
    scaled[col] = scaler.fit_transform(data[col])
    return(scaled)
cols = ["price","area"]
scaled_data = scale(data,cols)
scaled_data.head()

df_end = pd.get_dummies(data,columns=["location"]).astype(int)
print(df_end.head())
corr = df_end.corr()
corr.loc["price"]  #therefore the location of rural and sub urban areas are irrelevant

x = df_end[["area","bedrooms","location_Urban"]]
y = df_end["price"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
rmse = mse **0.5
print("\nRoot mean squared:",rmse)

r2 = r2_score(y_test,y_pred)
print("\nR-squared value:",r2)

#hence the model appears to be an okay fit, not too-good and not poor, hence not too reliable


# In[ ]:




