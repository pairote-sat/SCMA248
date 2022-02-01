#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

def load_data():
    return pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/ClaimsExperienceData.csv')
claims = load_data()
claims.head()


# In[ ]:


claims['NumClaims'] = claims['Freq']


# In[34]:


#claims.pivot_table(index = 'Freq', columns = 'Year', values = ['PolicyNum','NumClaims','PolicyNum'], aggfunc = {'PolicyNum':'count','NumClaims':'sum'}).rename(columns={'PolicyNum':'NumPolicies'})

claims.pivot_table(index = 'Freq', columns = 'Year', values = ['PolicyNum','NumClaims'], aggfunc = {'PolicyNum':['count',lambda x: x.count()],'NumClaims':'sum'})


# In[96]:


output = claims.pivot_table(index = 'Freq', columns = 'Year', values = 'PolicyNum', aggfunc = {'PolicyNum':lambda x: x.count()},margins=False)


# In[97]:


output.head()


# In[98]:


output.columns


# In[99]:


def calPercent(df,colName):
    name = 'Proportion_' + str(colName)
    df[name]=100*df[colName]/(df[colName].sum())
    return df


# In[101]:


calPercent(output,2010).head()


# In[ ]:




