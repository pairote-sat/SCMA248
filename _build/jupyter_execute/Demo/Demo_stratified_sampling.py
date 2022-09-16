#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import StratifiedKFold


# In[2]:


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[9, 10], [11,12]])
y = np.array([0, 0, 0, 1, 1, 1])

skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X, y):
    print("TRAIN INDEX:", train_index, "TEST INDEX:", test_index)


# In[3]:


X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


# In[4]:


X_train


# In[ ]:




