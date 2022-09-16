#!/usr/bin/env python
# coding: utf-8

# # Pandas’ groupby explained in detail

# In[1]:


import pandas as pd

order_leads = pd.read_csv(
    'https://raw.githubusercontent.com/FBosler/Medium-Data-Exploration/master/order_leads.csv',
    parse_dates = [3]
)
sales_team = pd.read_csv(
    'https://raw.githubusercontent.com/FBosler/Medium-Data-Exploration/master/sales_team.csv',
    parse_dates = [3]
)
df = pd.merge(
  order_leads,
  sales_team,
  on=['Company Id','Company Name']
)
df = df.rename(
  columns={'Order Value':'Val','Converted':'Sale'}
)


# In[29]:


print(order_leads.shape)
print(order_leads.columns)
print(order_leads[4600:4601])


# In[30]:


print(sales_team.shape)
print(sales_team.columns)
print(sales_team[4600:4601])


# In[19]:


df.shape


# In[31]:


df[4600:4601]


# In[34]:


df[df['Company Name']=='Dimensional Nitrogen']


# In[ ]:




