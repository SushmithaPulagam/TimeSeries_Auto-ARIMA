#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Installing pmdarima package
get_ipython().system(' pip install pmdarima')


# In[3]:


# Importing auto_arima 
from pmdarima.arima import auto_arima


# In[4]:


#Read the sales dataset
sales_data = pd.read_csv("Champagne Sales.csv")


# In[5]:


sales_data.head()


# In[6]:


#Make sure there are no null values at the end of the dataset
sales_data.tail()


# In[7]:


#Check the datatypes
sales_data.dtypes


# In[8]:


#Convert the month column to datetime
sales_data['Month']=pd.to_datetime(sales_data['Month'])


# In[9]:


#Recheck the datatypes
sales_data.dtypes


# In[10]:


#Set the index of the Month 
sales_data.set_index('Month',inplace=True)


# In[11]:


sales_data.head()


# In[12]:


# To understand the pattern
sales_data.plot()


# In[13]:


#Testing for stationarity
from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(sales_data)


# In[14]:


#Spliting the dataset into train and test
train = sales_data[:85]
test = sales_data[-20:]


# In[15]:


train.tail()


# In[16]:


test.head()


# In[17]:


plt.plot(train)
plt.plot(test)


# In[18]:


arima_model =  auto_arima(train,start_p=0, d=1, start_q=0, 
                          max_p=5, max_d=5, max_q=5, start_P=0, 
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True, 
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 50 )


# In[19]:


#Summary of the model
arima_model.summary()


# In[20]:


prediction = pd.DataFrame(arima_model.predict(n_periods = 20),index=test.index)
prediction.columns = ['predicted_sales']
prediction


# In[21]:


plt.figure(figsize=(8,5))
plt.plot(train,label="Training")
plt.plot(test,label="Test")
plt.plot(prediction,label="Predicted")
plt.legend(loc = 'Left corner')
plt.show()


# In[22]:


from sklearn.metrics import r2_score
test['predicted_sales'] = prediction
r2_score(test['Champagne sales'], test['predicted_sales'])


# In[ ]:




