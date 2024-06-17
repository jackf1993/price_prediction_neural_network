#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from keras.models import load_model


# In[9]:


model = load_model('trained_model.h5')

X = pd.read_csv('proposed_new_product.csv').values

prediction = model.predict(X)

prediction = prediction[0][0]

prediction = prediction + 0.153415
prediction = prediction / 0.000004

print("Earnings Prediction for Proposed Product - ${}".format(prediction))


# In[ ]:




