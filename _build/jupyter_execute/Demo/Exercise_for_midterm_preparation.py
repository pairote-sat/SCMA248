#!/usr/bin/env python
# coding: utf-8

# # Exercise for midterm preparation
# Instructions:
# 
# 1. This exercise contains 3 questions:
# 
# * 2 questions about aggregating and summarising time series data, and 
# 
# * 1 question about calculating Non-Cumulative into a new column from cumulative values.
# 
# 2. Try to complete these exercise problems  by yourself.
# 
# 3. Solutions to the problems are given below.

# In[1]:


import pandas as pd


# ## Working with datetime in Pandas DataFrame
# 
# In Pandas DataFrame, working with datetime
# 
# We will go over the following frequent datetime issues in this part, which should get you started with data analysis.
# 
# 1. Convert dates and times from strings.
# 
# 2. Create a datetime by combining numerous columns.
# 
# 3. Get the year's week, day of the week.

# ### Convert dates and times from strings.
# 
# To convert strings to datetime, Pandas comes with a built-in method named to `datetime()`. Let's look at a few examples.
# 
# #### With the default parameters
# 
# Without any further parameters, Pandas to datetime() can convert any valid date string to a datetime. Consider the following scenario:

# In[2]:


df = pd.DataFrame({'date': ['3/09/2022', '3/10/2022', '3/11/2022'],
                   'value': [2, 4, 8]})
df['date'] = pd.to_datetime(df['date'])
df


# ### Format to suit your needs
# 
# Your strings may be in a unique format, such as YYYY-DD-MM HH:MM:SS, for example. The `format` argument in Pandas to `datetime()` allows you to pass a custom format:

# In[3]:


df = pd.DataFrame({'date': ['2022-09-3', '2022-10-3', '2022-11-3'],
                   'value': [2, 4, 8]})

df['date'] = pd.to_datetime(df['date'], format="%Y-%d-%m")
df


# In[4]:


df = pd.DataFrame({'date': ['2022 09 3', '2022 10 3', '2022 11 3'],
                   'value': [2, 3, 4]})

df['date'] = pd.to_datetime(df['date'], format="%Y %d %m")
df


# #### Create a datetime by combining multiple columns.
# 
# The function `to_datetime()` can also be used to create a datetime from a collection of columns. The keys (column labels) can be abbreviations of ['year','month', 'day'].

# In[5]:


df = pd.DataFrame({'year': [2022, 2021],
                   'month': [1, 2],
                   'day': [15, 16]})
df['date'] = pd.to_datetime(df)
df


# #### Determine the year, month, and day.
# 
# The built-in characteristics `dt.year`, `dt.month`, and `dt.day` are used to get the year, month, and day from a Pandas datetime object.

# In[6]:


df = pd.DataFrame({'date': ['2022 09 3', '2022 10 3', '2022 11 3'],
                   'value': [2, 3, 4]})

df['date'] = pd.to_datetime(df['date'], format="%Y %d %m")

df['year']= df['date'].dt.year
df['month']= df['date'].dt.month
df['day']= df['date'].dt.day
df


# We are going to use data from Yahoo Finance databases. The data set consists,
# 
# * Open and close are the prices at which a stock began and ended trading in the same period. 
# 
# * Volume is the total amount of trading activity.

# We load the data and also use `parse_dates= ['Date']`
# to specify a list of date columns.

# In[7]:


url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/yahoo_stock.csv'
df = pd.read_csv(url)

#df = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/yahoo_stock.csv',parse_dates = ['Date'])


# In[19]:


df.info()


# **Exercise** write a python code to calculate the average yearly high, low, open, close and volumn.

# In[56]:


# Add your code here:






# **Exercise** write a python code to calculate the average quarterly high, low, open, close and volumn.

# In[56]:


# Add your code here:






# ## Calculating Non-Cumulative into a new column from cumulative values 

# Suppose that we have a series of cumulative values with a timestamp. We would like to get Python to show the non-cumulative values as shown below in the last column.

# In[48]:


df = pd.DataFrame({'date': ['3/09/2022', '3/10/2022', '3/11/2022', '3/12/2022'],
                   'cumulative_value': [2, 4, 8, 15]})
df['date'] = pd.to_datetime(df['date'])
df


# In[44]:


df['noncumulative_value'] = [2,2,4,7]
df


# **Exercise** Write a python code to create a column of non-cumulative values shown above.

# In[47]:


# Here we first drop the noncumulative_value column.

df.drop(['noncumulative_value'],axis = 1,inplace=True)
df


# In[56]:


# Add your code here:






# In[ ]:





# In[ ]:





# ## Solutions to exercise problems

# **Exercise** write a python code to calculate the average yearly high, low, open, close and volumn.

# **Solution** we can use the `pd.Grouper` function and the updated agg function to aggregatie and summarise data.

# #### pd.Grouper
# 
# The function `pd.Grouper` is useful when working with time-series data.
# 
# ##### Year-based grouping
# 
# We will use 
# 
# `pd.Grouper(key=INPUT COLUMN>, freq=DESIRED FREQUENCY>)`
# 
# in the following example to group our data depending on the supplied frequency for the specified column. 
# 
# The frequency in our situation is 'Y,' and the relevant column is 'Date.'

# In[25]:


df.groupby(pd.Grouper(key='Date',freq='Y')).mean()


# In[24]:


((df[df['Date'] <= '2015-12-31'])['Close']).mean()


# #### Quarter or other frequency grouping
# 
# Different standard frequencies, such as 'D','W','M', or 'Q', can be used instead of 'Y.'

# **Exercise** write a python code to calculate the average quarterly high, low, open, close and volumn.

# In[26]:


df.groupby(pd.Grouper(key='Date',freq='Q')).mean()


# **Exercise** Write a python code to create a column of non-cumulative values shown above.

# In[47]:


# Here we first drop the noncumulative_value column.

df.drop(['noncumulative_value'],axis = 1,inplace=True)
df


# To get the non-cumulative values, we simply take the time lag of numbers in the cumulative column using the `shift` function and then calculating the difference between cumulative and lag.

# In[49]:


df.head()


# In[54]:


df['lag'] = df.cumulative_value.shift(1).fillna(0)
df['noncumulative_value'] = df.cumulative_value - df.lag

df


# In[ ]:




