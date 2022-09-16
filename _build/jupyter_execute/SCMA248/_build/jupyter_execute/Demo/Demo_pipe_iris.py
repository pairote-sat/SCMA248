#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris

x = load_iris()
x = pd.DataFrame(x.data, columns=x.feature_names)

def remove_units(df):
    df.columns = pd.Index(map(lambda x: x.replace(" (cm)", ""), df.columns))
    return df

def length_times_width(df):
    df['sepal length*width'] = df['sepal length'] * df['sepal width']
    df['petal length*width'] = df['petal length'] * df['petal width']

x.pipe(remove_units).pipe(length_times_width)
x


# In[2]:


#gapminder = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/gapminder_full.csv')

url = 'https://raw.githubusercontent.com/STLinde/Anvendt-Statistik/main/gapminder_full.csv'
gapminder = pd.read_csv(url)
gapminder['year'].unique()


# In[9]:


cond = gapminder['year']==1957
gapminder.loc[cond,]


# In[10]:


#def replace_age_na(x_df, fill_map):
#    cond=x_df['Age'].isna()
#    res=x_df.loc[cond,'Pclass'].map(fill_map)
#    x_df.loc[cond,'Age']=res
#    return x_df


def year_filter(df,year_selected):
    cond=df['year']==year_selected
    return df.loc[cond,]


# In[ ]:


p = ggplot(data = gapminder1957) + aes(x = 'gdp_cap', y = 'life_exp', size = 'population') + geom_point(alpha = 0.8)

(
p + xlim(0,20000)
)


# In[1]:


#year_filter(gapminder,1957)
from plotnine import *

p = gapminder.pipe(year_filter,2007).pipe(ggplot) 

( p +
  aes(x = 'gdp_cap', y = 'life_exp', size = 'population') + geom_point(alpha = 0.8)
)


# In[ ]:


gapminder.pipe

