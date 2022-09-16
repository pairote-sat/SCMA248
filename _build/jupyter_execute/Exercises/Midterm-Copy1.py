#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from plotnine import *


# In[3]:


path = '/Users/Kaemyuijang/SCMA248/Data/us-counties.csv'

#df = pd.read_csv(path, parse_dates=True, index_col = 'date')

df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')


# In[4]:


df.info()


# In[5]:


df.index


# In[47]:


df['lag'] = df.cases.shift(1).fillna(0)
df['daily_cases'] = df.cases - df.lag


# In[51]:


df['lag_deaths'] = df.deaths.shift(1).fillna(0)
df['daily_deaths'] = df.deaths - df.lag_deaths


# In[57]:


df


# In[58]:


df_Snohomish = df.query('county == "Snohomish"')


df_Snohomish['lag'] = df_Snohomish.cases.shift(1).fillna(0)
df_Snohomish['daily_cases'] = df_Snohomish.cases - df_Snohomish.lag

(
    ggplot(df_Snohomish) + aes(x = df_Snohomish.index, y = 'cases') + geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[59]:


(
    ggplot(df_Snohomish) + aes(x = df_Snohomish.index, y = 'daily_cases') + geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[81]:


df_Snohomish


# In[69]:


df_Snohomish.daily_cases[df_Snohomish.daily_cases < 0]


# In[72]:


df_Snohomish.loc['2022-02-01':'2022-02-18']


# In[17]:


df.query('state=="Washington"').head(20)


# In[ ]:


df.groupby('county')


# In[80]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


path = '/Users/Kaemyuijang/SCMA248/Data/us-counties.csv'

#df = pd.read_csv(path, parse_dates=True, index_col = 'date')

df = pd.read_csv(path, parse_dates=['date'])


# In[85]:


df_Snohomish = df.query('county == "Snohomish"')

df_Snohomish['lag'] = df_Snohomish.cases.shift(1).fillna(0)
df_Snohomish['daily_cases'] = df_Snohomish.cases - df_Snohomish.lag


# In[88]:


print(df)
df_Snohomish


# In[90]:


pd_merge = pd.merge(df,df_Snohomish, how = 'left', on = ['date','county'])


# In[95]:


pd_merge.query('county == "Snohomish"')


# In[ ]:





# In[ ]:





# In[ ]:





# In[111]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[112]:


df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')

df = df.query('county in ["Snohomish","Cook"]')
#df = df.query('county in ["Snohomish"]')


# In[39]:


df.loc['2022-02-01':'2022-02-18']


# In[57]:


# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

pd.options.mode.chained_assignment = None


# In[58]:


df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    print(temp)
    
    plt.plot(temp.index,temp.daily_cases)
    #print(temp.query('county == @ct'))
    #df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)


# In[105]:


df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    
    rolling = temp.daily_cases.rolling(7, center=True)
    #print(rolling.mean().shape)
    #print(temp.shape)
    
    temp['rolling_mean'] = rolling.mean()
    #print(temp)
    
    df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)

(
    ggplot(df_new) + aes(x = df_new.index, y = 'daily_cases') + geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) + 
    facet_wrap('fips')
)


# In[106]:


df_new.loc[df_new['fips'] == 13075.]['2022-02-01':'2022-02-18']


# In[110]:


#df_new13075 = df_new.loc[df_new['fips'] == 13075.]['2021-07-01':'2022-02-18']

df_new13075 = df_new.query('fips == 13075.')

print(df_new13075.head(10))
(
    ggplot(df_new13075.query('fips == 13075.')) +
    geom_line(aes(x = df_new13075.index, y = 'daily_cases'), alpha = 0.5) +
    geom_line(aes(x = df_new13075.index, y = 'rolling_mean'), color='red',alpha=0.8) +
        theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[283]:


df_new17031 = df_new.query('fips == 17031.')

ggplot(df_new17031.query('fips == 17031.')) + aes(x = df_new17031.index, y = 'daily_cases') + geom_line()


# In[ ]:





# In[ ]:





# In[117]:


df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')

df = df.query('state in ["Washington"]')
#df = df.query('county in ["Snohomish"]')

print(df.county.unique())


# In[118]:


df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    
    rolling = temp.daily_cases.rolling(7, center=True)
    #print(rolling.mean().shape)
    #print(temp.shape)
    
    temp['rolling_mean'] = rolling.mean()
    #print(temp)
    
    df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)

(
    ggplot(df_new) + aes(x = df_new.index, y = 'daily_cases') + geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) + 
    facet_wrap('fips')
)


# In[126]:


fips_id = 53023.0

df_subset = df_new.query('fips==@fips_id')

(
    ggplot(df_subset) + geom_line(aes(x = df_subset.index, y = 'daily_cases'), alpha = 0.3) +    
    geom_line(aes(x = df_subset.index, y = 'rolling_mean'),color='red', alpha = 0.8) +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[ ]:





# In[ ]:





# In[260]:


df_new = pd.DataFrame()

for ct in df['county'].unique():
    print(ct)
    temp = df.query('county == @ct')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    print(temp)
    
    plt.plot(temp.index,temp.daily_cases)
    #print(temp.query('county == @ct'))
    #df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)
  


# In[241]:


test = df_new[df_new['county']=="Cook"]

print(test)

(
    ggplot(test) + aes(x = test.index, y = 'daily_cases') + geom_point() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[190]:


df


# In[120]:


pd_merge


# In[117]:


df.query('county == @name')


# In[174]:


df_merged = df

ct = "Cook"
#print(ct)
temp = df_merged.query('county == @name')
#print(temp)
temp['lag'] = temp.cases.shift(1).fillna(0)
temp['daily_cases'] = temp.cases - temp.lag
#print(temp[['county','daily_cases']])
#print(temp['county','daily_cases'])
df_merged = pd.merge(df_merged,temp, how = 'left', on = ['date','county','state','fips','cases','deaths'])
print(df_merged)


# In[162]:


temp.head()


# In[151]:


pd.merge(df_merged,temp, how = 'left', on = ['date','county','state','fips','cases','deaths'])


# In[ ]:




