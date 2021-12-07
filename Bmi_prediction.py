#Required libraries
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

#reading data from a file
df=pd.read_csv("BMI.csv")

#change gender column values

#considering Gender, Height and weight independent variables
x=df[["Height","Weight"]]

#dependent variable Bmi index 
y=df["Index"]

#split and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#using QuadraticDiscriminantAnalysis
regr=QuadraticDiscriminantAnalysis()
regr.fit(X_train, y_train)

#predict the Bmi Index
a=0
index=regr.predict([[195,104]])
print(index)

#print the accuracy of the model
print(regr.score(X_test,  y_test))

l=["Extremely weak","Weak","Normal","Overweight","Obesity","Extreme obesity"] 
for i in index:
 print(l[i])
 
pickle.dump(regr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


