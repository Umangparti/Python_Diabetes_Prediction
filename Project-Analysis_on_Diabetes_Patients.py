#!/usr/bin/env python
# coding: utf-8

# # Project-Analysis_on_Diabetes_Patients

# In[1]:


'''
Hello Everyone
I am Umang
I am going to Analyse this data on Diabetes Patients as my Project
Submitted to: Meriskill
'''


# In[2]:


'''
This dataset is originally from the National Institute of Diabetes 
and Digestive and Kidney Diseases. The objective of the dataset is 
to diagnostically predict whether a patient has diabetes based on 
certain diagnostic measurements included in the dataset. Several 
constraints were placed on the selection of these instances from a 
larger database. In particular, all patients here are females at 
least 21 years old of Pima Indian heritage.
'''


# In[3]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[4]:


# Loading the Dataset
patient_data = pd.read_csv(r"C:\Users\umang\Desktop\meriskill intern\project 2\diabetes.csv")


# # Exploring the Dataset

# In[5]:


print(patient_data)


# In[6]:


patient_data.columns


# In[7]:


patient_data.info()


# In[8]:


print(patient_data.describe())


# In[9]:


# We have data of 768 patients
# The numerical columns of Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
# BMI, DiabetesPredigreeFunction, and Age are independent variables
# The outcome column is a dependent variable. It shows whether the person has diabetes.
# In outcome column 1 ='Person has diabetes' and 0='Person not diabetic'


# # Cleaning the dataset

# In[10]:


patient_data.isnull().sum()


# In[11]:


patient_data.duplicated().sum()


# In[12]:


# No Null values in our data
# No duplicate values in our data


# # Correlation Matrix

# In[13]:


correlation = patient_data.corr()
print(correlation)


# In[14]:


#creating a correlation heatmap
sns.heatmap(correlation, xticklabels = correlation.columns, 
            yticklabels = correlation.columns, annot = True)
plt.show()


# In[15]:


# We notice that there is 0.54 positive correlation between Pregnancies and Age
# Showing that higher the Age, more the pregnancies.
# We also notice that all the factors affect the outcome of being Diabetic Positively
# Though their effect varies but more of these factors more chances of being diabetic
# Glucose and Diabetes are positively correlated with 0.47
# Factors like BMI, Pregnancies, and Age are correlated with range 0.2 to 0.3


# # Training and Predicting the Model

# In[16]:


x = patient_data.drop("Outcome", axis =1)
y = patient_data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)


# In[17]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


# In[18]:


# Predicting the model
prediction = model.predict(X_test)
print(prediction)


# In[19]:


# Checking the accuracy
accuracy = accuracy_score(prediction,Y_test)
print(accuracy)


# In[20]:


# We can say that our predictive model created is 81.16% accurate

