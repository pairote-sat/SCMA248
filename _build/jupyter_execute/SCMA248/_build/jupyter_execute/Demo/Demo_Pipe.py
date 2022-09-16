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
    return pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/train.csv')
df = load_data()
df.head()


# In[5]:


sns.heatmap(df.isnull(), 
            yticklabels=False, 
            cbar=False, 
            cmap='viridis')


# 1. Split Name into first name and second name
# 
# Let’s create a function split_name(), which takes in a DataFrame as input and returns a DataFrame.

# In[6]:


def split_name(x_df):
    def split_name_series(string):
        firstName, secondName=string.split(', ')
        return pd.Series(
            (firstName, secondName),
            index='firstName secondName'.split()
        )
    # Select the Name column and apply a function
    res=x_df['Name'].apply(split_name_series)
    x_df[res.columns]=res
    return x_df


# In[7]:


res=(
    load_data()
    .pipe(split_name)
)
res.head()


# 2. For Sex, substitute value male with M and female with F
# 
# Let’s create a function substitute_sex(), which takes in a DataFrame as input and returns a DataFrame.

# In[8]:


def substitute_sex(x_df):
    mapping={'male':'M','female':'F'}
    x_df['Sex']=df['Sex'].map(mapping)
    return x_df


# In[9]:


res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
)
res.head()


# 3. Replace the missing Age with some form of imputation
# 
# We would like to replace the missing Age with some form of imputation. One way to do this is by filling in the mean age of all the passengers. However, we can be smarter about this and check the average age by passenger class. For example:

# In[10]:


sns.boxplot(x='Pclass',
            y='Age',
            data=df,
            palette='winter')


# In[11]:


pclass_age_map = {
  1: 37,
  2: 29,
  3: 24,
}
def replace_age_na(x_df, fill_map):
    cond=x_df['Age'].isna()
    res=x_df.loc[cond,'Pclass'].map(fill_map)
    x_df.loc[cond,'Age']=res
    return x_df


# In[12]:


res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
    .pipe(replace_age_na, pclass_age_map)
)
res.head()


# In[13]:


sns.heatmap(res.isnull(), 
            yticklabels=False, 
            cbar=False, 
            cmap='viridis')


# 4. Convert ages to groups of age ranges: ≤12, Teen (≤18), Adult (≤60), and Older (>60)
# 
# Let’s create a function create_age_group(), which takes a DataFrame as input and returns a DataFrame.

# In[14]:


def create_age_group(x_df):
    bins=[0, 13, 19, 61, sys.maxsize]
    labels=['<12', 'Teen', 'Adult', 'Older']
    ageGroup=pd.cut(x_df['Age'], bins=bins, labels=labels)
    x_df['ageGroup']=ageGroup
    return x_df


# In[15]:


res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
    .pipe(replace_age_na, pclass_age_map)
    .pipe(create_age_group)
)
res.head()


# In[ ]:




