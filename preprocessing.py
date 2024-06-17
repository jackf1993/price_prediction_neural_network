#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential    
from keras import layers


# In[11]:


# load data
train_data_df = pd.read_csv('sales_data_training.csv')
test_data_df = pd.read_csv('sales_data_test.csv')


# In[13]:


# scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training = scaler.fit_transform(train_data_df)
scaled_testing = scaler.fit_transform(test_data_df)

print('Note: total_earnings values were scaled by muliplying by {:10f} and adding {:.6f}'.format(scaler.scale_[8],scaler.min_[8]))


# In[19]:


scaled_training_df = pd.DataFrame(scaled_training, columns=train_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

scaled_training_df.to_csv('sales_data_training_scaled.csv',index = False)
scaled_testing_df.to_csv('sales_data_testing_scaled.csv')


# In[ ]:




