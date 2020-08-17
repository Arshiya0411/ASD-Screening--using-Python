#!/usr/bin/env python
# coding: utf-8

# In[3]:
'''
Project: Autism Screening using a Logistic and Decision Tree model
Author: Arshiya Verma

'''

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn
import seaborn as sns


# Reading the data as a pandas data file


data_autism=pd.read_csv("/Users/arshiya/Desktop/Rutgers/Spring_20/Python_DS/Project/autism-screening.csv")
print(data_autism.head())
print(data_autism.info)
print(data_autism.shape)

#Exploring and Cleaning the data
print("The summary of data is")
print(data_autism.describe())
#Remove missing values
data_autism.replace("?",np.nan,inplace=True)
print("The summary of data without missing values")
print(data_autism.describe())
print(data_autism.head(20))
data_p=data_autism
print(data_p.dropna(inplace=True))
print(data_autism.columns)

#Converting the yes/no into 0/1 Boolean 

sex=pd.get_dummies(data_autism['gender'],drop_first=True)
jaund=pd.get_dummies(data_autism['jundice'],drop_first=True,prefix="Had_jaundice")
rel_autism=pd.get_dummies(data_autism['austim'],drop_first=True,prefix="Rel_had")
detected=pd.get_dummies(data_autism['Class/ASD'],drop_first=True,prefix="Detected")
data_autism=data_autism.drop(['gender','jundice','austim','Class/ASD'],axis=1)
data_featured=pd.concat([data_autism,sex,jaund,rel_autism,detected],axis=1)
data_featured.head()
data_featured.dropna
print(data_featured)

#Exploring the data with graphs

sns.countplot(x='Detected_YES',data=data_featured)
plt.show()
sns.countplot(x='Detected_YES',hue="Had_jaundice_yes",data=data_featured)
plt.show()
sns.countplot(x='Detected_YES',hue="m",data=data_featured)
plt.show()
sns.countplot(x='Detected_YES',hue="Rel_had_yes",data=data_featured)
plt.show()
sns.jointplot(x="result",y="Detected_YES",data=data_featured)
plt.show()

#print(data_autism.dtypes)
data_featured=data_featured.fillna(data_autism.mode().iloc[0])
data_autism = data_autism.fillna(data_autism.mode().iloc[0])
#print(data_autism.head(20))
#print(data_autism.columns)
#print(data_featured.columns)



X=data_featured[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'result', 'm',
       'Had_jaundice_yes']]
y=data_featured['Detected_YES']


corr = data_autism.corr()
plt.figure(figsize = (15,15))
sns.heatmap(data = corr, annot = True, square = True, cbar = True)
plt.show()

plt.figure(figsize = (16,8))
sns.countplot(x = 'ethnicity', data = data_autism)
plt.show()


#Fitting Machine Learning Models
#Dividing the data into train and test set for Logistic Regression
X=data_featured[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'm',
       'Had_jaundice_yes']]
y=data_featured['Detected_YES']

#print(X)
#print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
'''
print(X_train)
print(X_test)
print(y_train)
print(y_test)
'''
#Fitting a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lgr_model=LogisticRegression()
fit1=lgr_model.fit(X_train,y_train)
print("The fitted model is")
print(lgr_model.get_params(lgr_model))


#Predicting using the Logistic Regression
pred=lgr_model.predict(X_test)
print("The predicted values are")
print(pred)

#Finding metrics for predicted model
from sklearn.metrics import classification_report
print("The metrics of Logistic Regression are")
print(classification_report(y_true=y_test,y_pred=pred))
accuracy=sklearn.metrics.accuracy_score(y_test, pred, normalize=True, sample_weight=None)
print("The accuract is ", accuracy)

#Decision Tree
print("Fitting a Decision Tree model")
from sklearn import tree
tree_model=tree.DecisionTreeClassifier()
tree_model=tree_model.fit(X_train,y_train)
print("The predicted values using decision treee are")
pred_tree=tree_model.predict(X_test)
print(pred_tree)
print("The summary of the data is")
print(classification_report(y_true=y_test,y_pred=pred_tree))
print("The accuracy is")
print(sklearn.metrics.accuracy_score(y_test, pred_tree, normalize=True, sample_weight=None))

