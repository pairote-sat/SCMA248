#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from scipy import stats
from scipy.stats import *

ksN = 100           # Kolmogorov-Smirnov KS test for goodness of fit: samples
ALPHA = 0.05        # significance level for hypothesis test


# In[2]:


# example: properties of the beta distribution

a, b = 2, 6

x = beta.rvs(a, b, size=1000)

fig, ax = plt.subplots(1, 1)
ax.hist(x, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()


# In[3]:


beta.mean(a,b)


# In[4]:


x.mean()


# In[5]:


beta.b


# In[6]:


norm.b


# In[ ]:




