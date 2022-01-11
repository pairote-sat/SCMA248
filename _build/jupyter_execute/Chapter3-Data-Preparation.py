#!/usr/bin/env python
# coding: utf-8

# # Data Preparation 
# 
# 
# 
# ## Data Preparation with Pandas
# 
# This chapter will show you how to use the pandas package to import and preprocess data. Preprocessing is the process of pre-analyzing data before converting it to a standard and normalized format.
# The following are some of the aspects of preprocessing:
# 
# * missing values
# * data normalization 
# * data standardization 
# * data binning
# 
# We'll simply be dealing with missing values in this session.
# 
# #### Importing data 
# 
# We will utilize the Iris dataset in this tutorial, which can be downloaded from the UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/iris.
# 
# In the pattern recognition literature, this is probably the most well-known database. Fisher's paper is considered a classic in the subject and is still cited frequently. (See, for example, Duda & Hart.) The data collection has three classes, each with 50 instances, each referring to a different species of iris plant. The three classes include 
# * Iris Setosa
# * Iris Versicolour
# * Iris Virginica.
# 
# To begin, use the pandas library to import data and transform it to a dataframe. 

# In[1]:


import pandas as pd


# In[2]:


iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, 
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

type(iris)


# Here we specify whether there is a header (`header`) and the variable names (using `names` and a list). 
# 
# The resulting object `iris` is a pandas DataFrame. 
# 
# We will print the first 10 rows and the last 10 rows of the dataset using the head(10) method to get an idea of its contents.

# In[4]:


iris.head(10)


# In[5]:


iris.tail(10)


# To get the names of the columns (the variable names), you can use `columns` method.

# In[6]:


iris.columns


# To extract the class column, you can simply use the following commands:

# In[7]:


iris['class']


# The Pandas Series is a one-dimensional labeled array that may hold any type of data (integer, string, float, python objects, etc.)

# In[8]:


type(iris['class'])


# In[9]:


iris.dtypes


# Tab completion for column names (as well as public attributes), `iris.<TAB>` , is enabled by default if you're using Jupyter.
# 
# For example, type `iris.` and then follow with the TAB key. Look for the `shape` attribute.
# 
# The `shape` attribute of pandas.DataFrame stores the number of rows and columns as a tuple (number of rows, number of columns).

# In[10]:


iris.shape


# In[11]:


iris.info


# In[12]:


import pandas as pd

values = {'dates':  ['20210305','20210316','20210328'],
          'status': ['Opened','Opened','Closed']
          }

demo = pd.DataFrame(values)


# In[109]:


demo


# In[111]:


demo['dates'] = pd.to_datetime(demo['dates'], format='%Y%m%d')


# In[112]:


demo


# In[98]:


demo.to_csv('demo_df.csv')


# ## Data selection
# 
# We'll concentrate on how to slice, dice, and retrieve and set subsets of pandas objects in general. Because Series and DataFrame have received greater development attention in this area, they will be the key focus.
# 
# The axis labeling information in pandas objects is useful for a variety of reasons:
# 
# * Data is identified (metadata is provided) using established indicators, which is useful for analysis, visualization, and interactive console display.
# 
# * Allows for both implicit and explicit data alignment.
# 
# * Allows you to access and set subsets of the data set in an intuitive way.
# 
# 
# Three different forms of multi-axis indexing are currently supported by pandas.
# 
# 1. The indexing operators [] and attribute operator. in Python and NumPy offer quick and easy access to pandas data structures in a variety of situations.
# 
# 2. `.loc` is mostly label-based, but it can also be used with a boolean array. .When the items are not found, .loc will produce a KeyError.
# 
# 3. `.iloc` works with an integer array (from 0 to length-1 of the axis), but it can also work with a boolean array.
# 
# Except for slice indexers, which enable out-of-bounds indexing, .iloc will throw IndexError if a requested indexer is out-of-bounds. (This is in line with the Python/NumPy slice semantics.)

# In this section we will use the Iris dataset. First we obtain the row and column names by using the following commands:

# In[17]:


print(iris.index)
print(iris.columns)


# In[18]:


iris.head()


# ### The indexing operators []
# 
# To begin, simply indicate the column and line (by using its index) you're interested in.
# 
# You can use the following command to get the sepal width of the fifth line (index is 4):

# In[20]:


iris['sepal_width'][4]


# **Note:** Be careful, because this is not a matrix, and you might be tempted to insert the row first, then the column. Remember that it's a pandas DataFrame, and the [] operator operates on columns first, then the element of the pandas Series that results."

# Sub-matrix retrieval is a simple procedure that requires only the specification of lists of indexes rather than scalars.

# In[26]:


iris['sepal_width'][0:4]


# In[30]:


iris[['petal_width','sepal_width']][0:4]


# In[35]:


iris['sepal_width'][range(4)]


# ### .loc()
# 
# You can use the `.loc()` method to get something similar to the other approach (as in a matrix) of obtaining data.

# In[41]:


iris.loc[4,'sepal_width']


# In[44]:


iris.loc[0:4,'sepal_width']


# In[54]:


iris.loc[range(4),['petal_width','sepal_width']]


# ### .iloc()
# 
# Finally, there is `.iloc()`, which is a fully optimized function that defines the positions (as in a matrix). It requires you to define the cell using the row and column numbers.

# In[60]:


iris.iloc[4,1]


# The following commands produce the same output as `iris.loc[0:4,'sepal_width']` and `iris.loc[range(4),['petal_width','sepal_width']]`

# In[62]:


iris.iloc[0:4,1]


# In[67]:


iris.iloc[range(4),[3,1]]


# **Note:** .loc, .iloc, and also [] indexing can accept a callable as indexer as illustrated from the following examples. The callable must be a function with one argument (the calling Series or DataFrame) that returns valid output for indexing.

# In[75]:


iris.loc[:,lambda df: ['petal_length','sepal_length']]


# In[70]:


iris.loc[lambda iris: iris['sepal_width'] > 3.5, :]


# ## Dealing with problematic data
# 
# 
# ### Problem in setting index in pandas DataFrame
# 
# It may happen that the dataset contains an index column. How to import it correctly with Pandas? 
# 
# We will use a very simple dataset, namely demo_df.csv, that contains an index column (this is just a counter and not a feature)."

# In[116]:


df1 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv')
print(df1.head())
df1.columns


# We want to specify that 'Unnamed: 0' is the index column  while loading this data set with the following command (with the parameter `index_col`):

# In[100]:


df1 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv', index_col = 0)
df1.head()


# The dataset is loaded and the index is correct after performing the command.
# 
# ### Convert Strings to Datetime
# 
# However, we see an issue right away: all of the data, including dates, has been parsed as integers (or, in other cases, as string). If the dates do not have a particularly unusual format, you can use the autodetection routines to identify the column that contains the date data. It works nicely with the following arguments in this case:"

# In[117]:


df2 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv', index_col = 0, parse_dates = ['dates'])
df2.head()


# Alternatively, you can apply the following command to convert the integers to datetime:
# 
# `pd.to_datetime(df['DataFrame Column'], format=specify your format)` 
# 
# Remember that the date format for our example is yyyymmdd.
# 
# The following is a representation of this date format `format =  '%Y%m%d'` 
# 
# See https://datatofish.com/strings-to-datetime-pandas/ for more details.

# In[121]:


df3 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv', index_col = 0)
print(df3.head())

df3['dates'] = pd.to_datetime(df3['dates'], format='%Y%m%d')


# In[122]:


df3.head()


# ### Missing values
# 
# We will concentrate on missing values, which is perhaps the most challenging data cleaning operation.
# 
# It's a good idea to have an overall sense of a data set before you start cleaning it. After that, you can develop a plan for cleaning the data.
# 
# To begin, I like to ask the following questions:
# 
# * What are the features?
# 
# * What sorts of data are required (int, float, text, boolean)?
#  
# * Is there any evident data missing (values that Pandas can detect)?
# * Is there any other type of missing data that isn't as clear (and that Pandas can't easily detect)?
# 
# Let's have a look at an example by using a small sample data namely property_data.csv. 
# 
# In what follows, we also specify that 'PID' (personal indentifier) is the index column while loading this data set with the following command (with the parameter index_col):
# 

# In[139]:


df = pd.read_csv('/Users/Kaemyuijang/SCMA248/property_data.csv', index_col = 0)
df


# We notice that the PID (personal identifiers) as the index name has a missing value, i.e. NaN  (not any number). We will replace this missing PID with 10105 and also convert from floats to integers. 

# In[137]:


rowindex = df.index.tolist()
rowindex[4] = 10105.0
rowindex = [int(i) for i in rowindex]

df.index = rowindex

print(df.loc[:,'ST_NUM'])


# Alternatively, one can use Numpy to produce the same result. Simply run the following commands. Here we use `.astype()` method to convert the type of an array. 

# In[181]:


df = pd.read_csv('/Users/Kaemyuijang/SCMA248/property_data.csv', index_col = 0)
df

import numpy as np
rowindex = df.index.to_numpy()

rowindex[4] = 10105.0

df.index = rowindex.astype(int)

print(df.loc[:,'ST_NUM'])


# Now I can answer my first question: what are features? The following features can be obtained from the column names:
# 
# * ST_NUM is the street number
# 
# * ST_NAME is the street name
# 
# * OWN_OCCUPIED: Is the residence owner occupied?
# 
# * NUM_BEDROOMS: the number of rooms

# We can also respond to the question, What are the expected types?
# 
# * ST_NUM is either a float or an int... a numeric type of some sort
# 
# * ST_NAME is a string variable.
# 
# * OWN_OCCUPIED: string; OWN_OCCUPIED: string; OWN _OCCUPIED N ("No") or Y ("Yes")
# 
# * NUM_BEDROOMS is a numeric type that can be either float or int.

# ### Standard Missing Values
# 
# So, what exactly do I mean when I say "standard missing values?" These are missing values that Pandas can detect.
# 
# Let's return to our initial dataset and examine the "Street Number" column.
# 
# There are an empty cell in the third row (from the original file). A value of "NaN" appears in the seventh row.
# 
# Both of these numbers are obviously missing. Let's see how Pandas handle these situations. We can see that Pandas filled in the blank space with "NaN".
# 
# We can confirm that both the missing value and "NA" were detected as missing values using the isnull() method. True for both boolean responses.

# In[184]:


df['ST_NUM'].isnull()


# Similarly, for the NUM_BEDROOMS column of the original CSV file, users manually entering missing values with different names "n/a" and "NA". Pandas also recognized these as missing values and filled with "NaN".

# In[185]:


df['NUM_BEDROOMS']


# ### Missing Values That Aren't Standard
# 
# It is possible that there are missing values with different formats in some cases.
# 
# There are two other missing values in this column of different formats
# 
# * na
# 
# * `--`
# 
# Putting this different format in a list is a simple approach to detect them. When we import the data, Pandas will immediately recognize them. Here's an example of how we might go about it.

# In[191]:


# Making a list of missing value types
missing_values = ["na", "--"]

df = pd.read_csv('/Users/Kaemyuijang/SCMA248/property_data.csv', index_col = 0, na_values = missing_values)

df


# ### Unexpected Missing Values
# 
# We have observed both standard and non-standard missing data so far. What if we have a type that is not expected?
# 
# For instance, if our feature is supposed to be a string but it's a numeric type, it's technically a missing value.
# 
# Take a look at the column labeled "OWN_OCCUPIED" to understand what I'm talking about.

# In[195]:


df['OWN_OCCUPIED']


# We know Pandas will recognize the empty cell in row seven as a missing value because of our prior examples.
# 
# The number 12 appears in the fourth row. This number type should be a missing value because the result for Owner Occupied should clearly be a string (Y or N).
# Because this example is a little more complicated, we will need to find a plan for identifying missing values. There are a few alternative routes to take, but this is how I'm going to go about it.
# 
# 1. Loop through The OWN OCCUPIED column.
# 
# 2. Convert the entry to an integer.
# 
# 3. If the entry may be transformed to an integer,  enter a missing value. 
# 
# 4. We know the number cannot be an integer if it cannott be an integer.

# In[219]:


df = pd.read_csv('/Users/Kaemyuijang/SCMA248/property_data.csv', index_col = 0)
df

import numpy as np
rowindex = df.index.to_numpy()

rowindex[4] = 10105.0

df.index = rowindex.astype(int)


# In[220]:


df


# In[221]:


# Detecting numbers 
cnt=10101
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1


# In[222]:


df['OWN_OCCUPIED']


# In the code, we loop through each entry in the "Owner Occupied" column. To try to change the entry to an integer, we use int(row).
# 
# If the value can be changed to an integer, we change the entry to a missing value using np.nan from Numpy.
# 
# On the other hand, if the value cannot be changed to an integer, we pass it and continue.
# 
# You will notice that I have used try and except ValueError. This is called exception handling, and we use this to handle errors.
# 
# If we tried to change an entry to an integer and it could not be changed, a ValueError would be returned and the code would terminate. To deal with this, we use exception handling to detect these errors and continue.
