#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd

from keras.models import Sequential as sq
from keras.layers import Dense
import tensorflow as tf
import keras
from keras import layers


# In[46]:


training_data_df = pd.read_csv('sales_data_training_scaled.csv')
testing_data_df = pd.read_csv('sales_data_testing_scaled.csv')
testing_data_df = testing_data_df.drop('Unnamed: 0', axis = 1)

X = training_data_df.drop('total_earnings',axis = 1)
y = training_data_df[['total_earnings']].values


# In[47]:


X_test = testing_data_df.drop('total_earnings', axis = 1)

y_test = testing_data_df[['total_earnings']].values


# In[77]:


model = keras.Sequential([
    keras.layers.Dense(50, input_dim = 9, activation = 'relu'),
    keras.layers.Dense(100,activation = 'relu'),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dense(1,activation = 'linear')
])


model.compile(optimizer ='adam',
              loss = 'mean_squared_error')

model.fit(X,y,epochs=50, shuffle = True, verbose = 2)

test_error_rate = model.evaluate(X_test,y_test)  
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


# In[75]:


model.save('trained_model.h5')
print('Model saved to disk.')


# In[ ]:




