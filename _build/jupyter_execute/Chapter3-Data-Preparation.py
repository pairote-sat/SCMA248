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

# In[11]:


iris.head(10)


# In[12]:


iris.tail(10)


# To get the names of the columns (the variable names), you can use `columns` method.

# In[13]:


iris.columns


# To extract the class column, you can simply use the following commands:

# In[14]:


iris['class']


# The Pandas Series is a one-dimensional labeled array that may hold any type of data (integer, string, float, python objects, etc.)

# In[15]:


type(iris['class'])


# In[16]:


iris.dtypes


# Tab completion for column names (as well as public attributes), `iris.<TAB>` , is enabled by default if you're using Jupyter.
# 
# For example, type `iris.` and then follow with the TAB key. Look for the `shape` attribute.
# 
# The `shape` attribute of pandas.DataFrame stores the number of rows and columns as a tuple (number of rows, number of columns).

# In[17]:


iris.shape


# In[18]:


iris.info


# In[19]:


import pandas as pd

values = {'dates':  ['20210305','20210316','20210328'],
          'status': ['Opened','Opened','Closed']
          }

demo = pd.DataFrame(values)


# In[20]:


demo


# In[21]:


demo['dates'] = pd.to_datetime(demo['dates'], format='%Y%m%d')


# In[22]:


demo


# In[23]:


demo.to_csv('demo_df.csv')


# ## Data Selection
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

# In[24]:


print(iris.index)
print(iris.columns)


# In[25]:


iris.head()


# ### The indexing operators []
# 
# To begin, simply indicate the column and line (by using its index) you're interested in.
# 
# You can use the following command to get the sepal width of the fifth line (index is 4):

# In[26]:


iris['sepal_width'][4]


# **Note:** Be careful, because this is not a matrix, and you might be tempted to insert the row first, then the column. Remember that it's a pandas DataFrame, and the [] operator operates on columns first, then the element of the pandas Series that results."

# Sub-matrix retrieval is a simple procedure that requires only the specification of lists of indexes rather than scalars.

# In[27]:


iris['sepal_width'][0:4]


# In[28]:


iris[['petal_width','sepal_width']][0:4]


# In[29]:


iris['sepal_width'][range(4)]


# ### .loc()
# 
# You can use the `.loc()` method to get something similar to the other approach (as in a matrix) of obtaining data.

# In[30]:


iris.loc[4,'sepal_width']


# In[31]:


# rows at index labels between 0 and 4 (inclusive)
# See https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different

iris.loc[0:4,'sepal_width']


# In[32]:


iris.loc[range(4),['petal_width','sepal_width']]


# ### .iloc()
# 
# Finally, there is `.iloc()`, which is a fully optimized function that defines the positions (as in a matrix). It requires you to define the cell using the row and column numbers.

# In[33]:


iris.iloc[4,1]


# The following commands produce the same output as `iris.loc[0:4,'sepal_width']` and `iris.loc[range(4),['petal_width','sepal_width']]`

# In[34]:


# rows at index locations between 0 and 4 (exclusive)
# See https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different

iris.iloc[0:4,1]


# In[35]:


iris.iloc[range(4),[3,1]]


# **Note:** .loc, .iloc, and also [] indexing can accept a callable as indexer as illustrated from the following examples. The callable must be a function with one argument (the calling Series or DataFrame) that returns valid output for indexing.

# In[36]:


iris.loc[:,lambda df: ['petal_length','sepal_length']]


# In[37]:


iris.loc[lambda iris: iris['sepal_width'] > 3.5, :]


# ## Dealing with Problematic Data
# 
# 
# ### Problem in setting index in pandas DataFrame
# 
# It may happen that the dataset contains an index column. How to import it correctly with Pandas? 
# 
# We will use a very simple dataset, namely demo_df.csv (the file can be download from my github repository), that contains an index column (this is just a counter and not a feature).

# In[38]:


import pandas as pd

# How to read CSV file from GitHub using pandas
# https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas

url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/demo_df'
df1 = pd.read_csv(url)
print(df1.head())
df1.columns


# In[39]:


## Uncomment these commands if the CSV dateset is stored locally.

# df1 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv')
# print(df1.head())
# df1.columns


# We want to specify that 'Unnamed: 0' is the index column  while loading this data set with the following command (with the parameter `index_col`):

# In[40]:


df1 = pd.read_csv(url, index_col = 0)
df1.head()


# The dataset is loaded and the index is correct after performing the command.
# 
# ### Convert Strings to Datetime
# 
# However, we see an issue right away: all of the data, including dates, has been parsed as integers (or, in other cases, as string). If the dates do not have a particularly unusual format, you can use the autodetection routines to identify the column that contains the date data. It works nicely with the following arguments when the data file is stored locally.

# In[41]:


# df2 = pd.read_csv('/Users/Kaemyuijang/SCMA248/demo_df.csv', index_col = 0, parse_dates = ['dates'])
# print(df2.head())
# df2.dtypes


# For the same dataset downloaded from Github, if a column or index contains an unparseable date, the entire column or index will be returned unaltered as an object data type. For non-standard datetime parsing, use pd.to_datetime after pd.read_csv:
# 
# `pd.to_datetime(df['DataFrame Column'], format=specify your format)` 
# 
# Remember that the date format for our example is yyyymmdd.
# 
# The following is a representation of this date format `format =  '%d%m%Y'` (or `format =  '%Y%m%d'`). 
# 
# See https://datatofish.com/strings-to-datetime-pandas/ for more details.

# In[42]:


df2 = pd.read_csv(url, index_col = 0, parse_dates = ['dates'])
print(df2)
df2.dtypes


# In[43]:


df2['dates'] = pd.to_datetime(df2['dates'], format='%d%m%Y')
print(df2)
df2.dtypes


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
# Let's have a look at an example by using a small sample data namely property_data.csv. The file can be obtained from Github: https://raw.githubusercontent.com/pairote-sat/SCMA248/main/property_data.csv.
# 
# In what follows, we also specify that 'PID' (personal indentifier) is the index column while loading this data set with the following command (with the parameter index_col):
# 

# In[10]:


url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/property_data.csv'

df = pd.read_csv(url, index_col = 0)
df


# We notice that the PID (personal identifiers) as the index name has a missing value, i.e. NaN  (not any number). We will replace this missing PID with 10105 and also convert from floats to integers. 

# In[11]:


rowindex = df.index.tolist()
rowindex[4] = 10105.0
rowindex = [int(i) for i in rowindex]

df.index = rowindex

print(df.loc[:,'ST_NUM'])


# Alternatively, one can use Numpy to produce the same result. Simply run the following commands. Here we use `.astype()` method to convert the type of an array. 

# In[46]:


df = pd.read_csv(url, index_col = 0)
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

# In[47]:


df['ST_NUM'].isnull()


# Similarly, for the NUM_BEDROOMS column of the original CSV file, users manually entering missing values with different names "n/a" and "NA". Pandas also recognized these as missing values and filled with "NaN".

# In[48]:


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

# In[49]:


# Making a list of missing value types
missing_values = ["na", "--"]

df = pd.read_csv(url, index_col = 0, na_values = missing_values)

df


# ### Unexpected Missing Values
# 
# We have observed both standard and non-standard missing data so far. What if we have a type that is not expected?
# 
# For instance, if our feature is supposed to be a string but it's a numeric type, it's technically a missing value.
# 
# Take a look at the column labeled "OWN_OCCUPIED" to understand what I'm talking about.

# In[50]:


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

# In[51]:


df = pd.read_csv(url, index_col = 0)
df

import numpy as np
rowindex = df.index.to_numpy()

rowindex[4] = 10105.0

df.index = rowindex.astype(int)


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

fig, ax = plt.subplots(figsize=(12, 4))

scaler = StandardScaler()
x_std = scaler.fit_transform(x)


# In[53]:


# Detecting numbers 
cnt=10101
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1


# In[54]:


df['OWN_OCCUPIED']


# In the code, we loop through each entry in the "Owner Occupied" column. To try to change the entry to an integer, we use int(row). If the value can be changed to an integer, we change the entry to a missing value using np.nan from Numpy. On the other hand, if the value cannot be changed to an integer, we pass it and continue.
# 
# You will notice that I have used try and except ValueError. This is called exception handling, and we use this to handle errors.
# 
# If we tried to change an entry to an integer and it could not be changed, a ValueError would be returned and the code would terminate. To deal with this, we use exception handling to detect these errors and continue.

# ## Data Manipulation
# 
# We have learned how to select the data we want, we will need to learn how to manipulate it. Using aggregation methods to work with columns or rows is one of the most straightforward things we can perform.
# 
# All of these functions always **return a number** when applied to a row or column. 
# 
# We can specify whether the function should be applied to 
# 
# * the rows for each column using the `axis=0` keyword on the function argument, or 
# 
# * the columns for each row using the `axis=1` keyword on the function argument.
# 
# | Function | Description |
# | ----------- | ----------- |
# | df.describe() | Returns a summary statistics of numerical column |
# | df.mean()  |  Returns the average of all columns in a dataset |
# df.corr()  |  Returns the correlation between columns in a DataFrame | 
# | df.count()  |  Returns the number of non-null values in each DataFrame column |
# | df.max()  |  Returns the highest value in each column |
# | df.min() | Returns the lowest value in each column |
# | df.median()  |  Returns the median in each column |
# | df.std()  | Returns the standard deviation in each column |

# In[77]:


iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, 
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

type(iris)


# In[78]:


iris.describe()


# You have the number of observations, their average value, standard deviation, minimum and maximum values, and certain percentiles for all numerical features" (25 percent, 50 percent, and 75 percent). This offers you a fair picture of how each feature is distributed.
# 
# The following command illustrate how to use the `max()` function.

# In[159]:


iris.max(axis = 0)


# We can perform operations on all values in rows, columns, or a subset of both. 
# 
# The following example shows how to find the standardized values of each column of the Iris dataset. We need to firstly drop the class column, which is a categorical variable using `drop()` function with `axis = 1`. 
# 
# Recall that the Z-scores and standardized values (sometimes known as standard scores or normal deviates) are the same thing. When you take a data point and scale it by population data, you get a standardized value. It informs us how distant we are from the mean in terms of standard deviations.

# In[260]:


iris_drop = iris.drop('class', axis = 1)


# In[205]:


print(iris_drop.mean())
print(iris_drop.std())


# In[244]:


def z_score_standardization(series):
    return (series - series.mean()) / series.std(ddof = 1)

#iris_normalized = iris_drop
#for col in iris_normalized.columns:
#    iris_normalized[col] = z_score_standardization(iris_normalized[col])

iris_normalized = {}
for col in iris_drop.columns:
    iris_normalized[col] = z_score_standardization(iris_drop[col])
    
iris_normalized = pd.DataFrame(iris_normalized)


# In[246]:


iris_normalized.head()


# Alternatively, the following commands produce the same result.

# In[248]:


del(iris_normalized)

iris_normalized = (iris_drop - iris_drop.mean())/iris_drop.std(ddof=1)
iris_normalized.head()


# Alternatively, we can standardize a Pandas Column with Z-Score Scaling using scikit-learn.

# In[252]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris_drop)

iris_scaled = scaler.fit_transform(iris_drop)
iris_scaled = pd.DataFrame(iris_scaled, columns=iris_drop.columns)
print(iris_scaled)


# **Note**  scikit-learn uses np.std which by **default is the population standard deviation** (where the sum of squared deviations are divided by the number of observations), while the sample standard deviations (where the denominator is number of observations - 1) are used by pandas. This is a correction factor determined by the degrees of freedom in order to obtain an unbiased estimate of the population standard deviation (ddof). Numpy and scikit-learn calculations utilize ddof=0 by default, whereas pandas uses ddof=1 (docs).
# 
# As a result, the above results are different. 
# 
# https://stackoverflow.com/questions/44220290/sklearn-standardscaler-result-different-to-manual-result

# In the next example, we simply can apply any binary arithmetical operation (+,-,*,/) to an entire dataframe.

# In[267]:


iris_drop**2


# Any function can be applied to a DataFrame or Series by passing its name as an argument to the `apply` method. For example, in the following code, we use the NumPy library's `floor` function to return the floor of the input, element-wise of each value in the DataFrame.

# In[271]:


import numpy as np

iris_drop.apply(np.floor)


# If we need to design a specific function to apply it, we can write an in-line function, commonly known as a $\lambda$-function. A $\lambda$-function is a function without a name. 
# 
# It is only necessary to specify the parameters it receives, between the lambda keyword and the colon (:). 
# 
# In the next example, only one parameter is needed, which will be the value of each element in the DataFrame.

# In[274]:


iris_drop.apply(lambda x: np.log10(x))


# ### Add A New Column To An Existing Pandas DataFrame
# 
# Adding new values in our DataFrame is another simple manipulation technique. This can be done directly over a DataFrame with the assign operator (`=`). 
# 
# For example, we can assign a Series to a selection of a column that does not exist to add a new column to a DataFrame. 
# 
# You should be aware that previous values will be overridden if a column with the same name already exists. 
# 
# In the following example, we create a new column entitled sepal_length_normalized by adding the standardized values of the sepal_length column.
# 

# In[285]:


iris['sepal_length_normalized'] = (iris['sepal_length'] - iris['sepal_length'].mean()) / iris['sepal_length'].std()
iris.head()


# In[333]:


iris['sepal_length_normalized'] = (iris['sepal_length'] - iris['sepal_length'].mean()) / iris['sepal_length'].std(ddof=0)
iris.head()


# Alternatively, we can use `concat` function to add a Series (s) to a Pandas DataFrame (df) as a new column with an arguement `axis = 1`. The name of the new column can be set by using `Series.rename` as in the following command:
# 
# `df = pd.concat((df, s.rename('CoolColumnName')), axis=1)`
# 
# 
# https://stackoverflow.com/questions/39047915/concat-series-onto-dataframe-with-column-name

# In[312]:


pd.concat([iris, (iris['sepal_length']-1).rename('new_name') ], axis = 1)


# We can now use the `drop` function to delete a column from the DataFrame; this removes the indicated rows if axis=0, or the indicated columns if axis=1. 
# 
# All Pandas functions that change the contents of a DataFrame, such as the drop function, return a duplicate of the updated data rather than overwriting the DataFrame. As a result, the original DataFrame is preserved. Set the keyword `inplace` to `True` if you do not wish to maintain the old settings. This keyword is set to `False` by default, which means that a copy of the data is returned.
# 
# The following commands remove the column, namely 'sepal_length_normalized', we have just added to the Iris dataset.

# In[323]:


iris.columns


# In[326]:


print(iris.drop('sepal_length_normalized', axis = 1).head())

print(iris.head())


# In[335]:


iris.drop('sepal_length_normalized', axis = 1, inplace = True)

print(iris.head())


# ### Appending a row to a dataframe and specify its index label
# 
# In this section, we will learn how to add/remove new rows and remove missing values.  We will use the dataset, namely property_data.csv. 
# 
# A general solution when appending a row to a dataframe and specify its index label is to create the row, transform the new row data into a pandas series, name it to the index you want to have and then append it to the data frame. Don't forget to overwrite the original data frame with the one with appended row. The following commands produce the required result.
# 
# See https://stackoverflow.com/questions/16824607/pandas-appending-a-row-to-a-dataframe-and-specify-its-index-label for more details.
# 
# See also https://www.geeksforgeeks.org/python-pandas-dataframe-append/ and
# https://thispointer.com/python-pandas-how-to-add-rows-in-a-dataframe-using-dataframe-append-loc-iloc/#3 .

# In[2]:


import pandas as pd

filepath = '/Users/Kaemyuijang/SCMA248/Data/property_data.csv'

df = pd.read_csv(filepath, index_col = 0)
df.head()


# In[39]:


url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/property_data.csv'

df = pd.read_csv(url, index_col = 0)
df.head()


# In[4]:


df.index


# In[5]:


#new_observation = {'ST_NUM': 555 , 'OWN_OCCUPIED': 'Y', 'NUM_BEDROOMS': 2,'NUM_BATH': 1,'SQ_FT': 200}

new_observation = pd.Series({'ST_NUM': 555 , 'OWN_OCCUPIED': 'Y', 'NUM_BEDROOMS': 2,'NUM_BATH': 1,'SQ_FT': 200}, name = 10110.0)


# In[6]:


new_observation


# In[7]:


df.append(new_observation)


# **Note** In case that, we do not define a name for our pandas series, i.e. we simply define `next_observation = pd.Series({'ST_NUM': 999 , 'OWN_OCCUPIED': 'Y', 'NUM_BEDROOMS': 2,'NUM_BATH': 1,'SQ_FT': 200})` without the flag `name = 10110.0` as an argument. We must set the `ignore_index` flag in the append method to `True`, otherwise the commands will produce an error as follows:

# In[10]:


next_observation = pd.Series({'ST_NUM': 999 , 'OWN_OCCUPIED': 'N', 'NUM_BEDROOMS': 5,'NUM_BATH': 4,'SQ_FT': 3500})


# In[11]:


print(next_observation)
print(df)


# In[12]:


# Without setting the ignore_index flag in the append method to True, this produces an eror.

df.append(next_observation)


# In[13]:


# Setting the ignore_index flag in the append method to True

df.append(next_observation,ignore_index=True)


# **Note** The resulting (new) DataFrame’s index is not same as original dataframe because ignore_index is passed as True in the `append()` function.

# The next complete example illustrates how to add multiple rows to dataframe.

# In[14]:


print(new_observation)
print(next_observation)


# In[15]:


listOfSeries = [new_observation,next_observation]

new_df = df.append(listOfSeries, ignore_index = True)


# Finally, to remove the row(s), we can apply the drop method once more. Now we must set the axis to 0 and specify the row index we want to remove. 

# In[16]:


new_df.drop([9,10], axis = 0, inplace = True)


# In[17]:


new_df


# By applying the `drop()` function to the result of the `isnull()` method, missing values can be removed. This has the same effect as filtering the NaN values, as explained above, but instead of returning a view, a duplicate of the DataFrame minus the NaN values is returned.

# In[ ]:


index = new_df.index[new_df['ST_NUM'].isnull()]

# Altenatively, one can obtain the index 
new_index = new_df['ST_NUM'].index[new_df['ST_NUM'].apply(np.isnan)]

print(new_df.drop(index, axis = 0))
print(new_df.drop(new_index, axis = 0))


# Instead of using the generic drop function, we can use the particular `dropna()` function to remove NaN values. We must set the `how` keyword to `any` if we wish to delete any record that includes a NaN value. We can use the `subset` keyword to limit it to a specific subset of columns. The effect will be the same as if we used the drop function, as shown below:
# 
# **Note** The parameter `how{‘any’, ‘all’}`, default ‘any’ determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
# 
# * 'any' : If any NA values are present, drop that row or column.
# 
# * 'all' : If all values are NA, drop that row or column.

# In[67]:


df.dropna(how = 'any', subset = ['ST_NUM'])


# In[69]:


# Setting subset keyword with a subset of columns

df.dropna(how = 'any', subset = ['ST_NUM','NUM_BATH'])


# If we wish to fill the rows containing NaN with another value instead of removing them, we may use the `fillna()` method and specify the value to use. If we only want to fill certain columns, we must pass a dictionary as an argument to the `fillna()` function, with the names of the columns as the key and the character to use for filling as the value.

# In[71]:


df.fillna(value = {'ST_NUM':0})


# In[73]:


df.fillna(value = {'ST_NUM': -1, 'NUM_BEDROOMS': -1 })


# ### Sorting
# 
# Sorting by columns is another important feature we will need while examining at our data. Using the sort method, we can sort a DataFrame by any column. We only need to execute the following commands to see the first five rows of data sorted in descending order (i.e., from the largest to the lowest values) and using the Value column:
# 
# Here we will work with the Iris dataset.

# In[80]:


print(iris.columns)


# In[84]:


iris.sort_values(by = ['sepal_length'], ascending = False)


# The following command sorts the sepal_lenghth column followed by the petal_length column. 

# In[85]:


iris.sort_values(by = ['sepal_length','petal_length'], ascending = False)


# **Note**
# 
# 1. The `inplace` keyword indicates that the DataFrame will be overwritten, hence no new DataFrame will be returned. 
# 
# 2. When `ascending = True` is used instead of `ascending = False`, the values are sorted in ascending order (i.e., from the smallest to the largest values).
# 
# 3. If we wish to restore the original order, we can use the `sort_index` method and sort by an index.

# In[88]:


# restore the original order

iris.sort_index(axis = 0, ascending = True, inplace = True)
iris.head()


# In[4]:


# Importing sales dataset for the next section

filepath = '/Users/Kaemyuijang/SCMA248/Data/SalesData.csv'

# pd.read_csv(filepath, header = None, skiprows = 1, names = ['Trans_no','Name','Date','Product','Units','Dollars','Location'])

# header = 0 means that the first row to use as the column names
sales = pd.read_csv(filepath, header = 0,  index_col = 0)

# https://stackoverflow.com/questions/25015711/time-data-does-not-match-format
sales['Date'] =  pd.to_datetime(sales['Date'], format='%d/%m/%Y')

sales.head()


# ### Grouping Data
# 
# 
# Another useful method for inspecting data is to group it according to certain criteria. For the sales dataset, it would be useful to categorize all of the data by location, independent of the year. The `groupby` function in Pandas allows us to accomplish just that. This function returns a special grouped DataFrame as a result. As a result, an aggregate function must be used to create a suitable DataFrame. As a result, **all values in the same group will be subjected to this function**.
# 
# For instance, in our scenario, we can get a DataFrame that shows the number (count) of the transactions for each location across all years by grouping by location and using the count function as the aggregation technique for each group. As a consequence, a DataFrame with locations as indexes and counting values of transactions as the column would be created:

# In[5]:


print(type(sales[['Location','Dollars']].groupby('Location')))

sales[['Location','Dollars']].groupby('Location').count()


# In[7]:


# a DataFrame with locations as indexes and mean of sales income  as 
# the column would be created

sales[['Location','Dollars']].groupby('Location').mean()


# In[10]:


sales[['Location','Dollars','Units']].groupby('Location').mean()


# ### Rearranging Data
# 
# Until now, our indices were merely a numerical representation of rows with little meaning. We can change the way our data is organized by redistributing indexes and columns for better data processing, which usually results in greater performance. Using the `pivot_table` function, we may rearrange our data. We can define which columns will be the new indexes, values, and columns.
# 
# 
# ####  Simplest Pivot table
# 
# An `index` is required even for the smallest pivot table. Let us utilize the location as our index in this case. It uses the `'mean'` aggregation function on all available numerical columns by default.

# In[124]:


sales.pivot_table(index='Location')


# To display multiple indexes, we can pass a list to index:

# In[125]:


sales.pivot_table(index=['Location','Product'])


# On the pivot table, the values to index are the keys to group by. To achieve a different visual representation, you can change the order of the values. For example, we can look at average values by grouping the region with the product category.

# In[126]:


sales.pivot_table(index=['Product','Location'])


# #### Specifying values and performing aggregation
# 
# The mean aggregation function is applied to all numerical columns by default, and the result is returned. Use the `values` argument to specify the columns we are interested in.

# In[129]:


sales.pivot_table(index=['Location'],
                         values = ['Dollars'])


# We can specify a valid string function to `aggfunc` to perform an aggregation other than mean, for example, a sum:

# In[132]:


sales.pivot_table(index=['Location'],
                         values = ['Dollars'], aggfunc = 'sum')


# `aggfunc` can be a dict, and below is the dict equivalent.

# In[133]:


sales.pivot_table(index=['Location'],
                         values = ['Dollars'], aggfunc = {'Dollars': 'sum'})


# `aggfunc` can be a list of functions, and below is an example to display the `sum` and `count`

# In[134]:


sales.pivot_table(index=['Location'],
                         values = ['Dollars'], aggfunc = ['sum','count'])


# #### Seeing break down using columns
# 
# If we would like to see sales broken down by product_category, the columns argument allows us to do that

# In[135]:


sales.pivot_table(index=['Location'],
                  values = ['Dollars'], 
                  aggfunc = 'sum',
                 columns='Product')


# **Note**  If there are missing values and we want to replace them, we could use `fill_value` argument, for example, to set `NaN` to a specific value.

# In[136]:


sales.pivot_table(index=['Location'],
                  values = ['Dollars'], 
                  aggfunc = 'sum',
                 columns='Product',
                 fill_value = 0)


# ### Exercise
# 
# Apply `pivot_table` (in new worksheets) to answer the following questions. 
# 
# 1. The number of sales transactions for the given salesperson
# 2. For the given salesperson, the total revenue by the given product
# 3. Total revenue generated by the given salesperson broken down by the given location
# 4. Total revenue by the salesperson and the given year

# ### Solutions to Exercise
# 
# 1. The number of sales transactions for the given salesperson
# 

# In[13]:


sales.columns


# In[33]:


sales.pivot_table(index=['Name'],
                 values=['Dollars'], aggfunc='count')


# In[34]:


sales.pivot_table(index=['Name'],
                 values=['Dollars'], aggfunc = 'count').sort_values(by = 'Dollars')


# 2. For the given salesperson, the total revenue by the given product

# In[52]:


sales.pivot_table(index='Name',
                 values='Dollars',
                  columns = 'Product', aggfunc = 'sum',  margins=True)


# The result above also show the total. In Panda pivot_table(), we can simply pass `margins=True`.

# 3. Total revenue generated by the given salesperson broken down by the given location

# In[50]:


sales.pivot_table(index=['Name','Location'],
                 values='Dollars', aggfunc = 'sum').head(10)


# 4. Total revenue by the salesperson and the given year. 
# 
# To generate a yearly report with Panda pivot_table(), here are the steps:
# 
# 1. Defines a groupby instruction using Grouper() with key='Date' and freq='Y'.
# 2. Applies pivot_table.
# 
# **Note** To group our data depending on the specified frequency for the specified column, we'll use `pd.Grouper(key=INPUT COLUMN>, freq=DESIRED FREQUENCY>)`. The frequency in our situation is 'Y' and the relevant column is 'Date.'
# 
# Different standard frequencies, such as 'D','W','M', or 'Q', can be used instead of 'Y.' Check out the following for a list of less common useable frequencies, https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.

# In[27]:


year_gp = pd.Grouper(key='Date',freq='Y')
print(year_gp)

sales.pivot_table(index='Name', columns=year_gp, values='Dollars',aggfunc='sum')


# Exercise
# 
# Apply pivot_table (in new worksheets) to answer the following questions.
# 
# 5. How many saleperson are there? Hint: use `groupby` to create a grouped DataFrame grouped by salepersons and then call `groups` on the grouped object, which will returns the list of indices for every group.

# In[13]:


grouped_person = sales.groupby('Name')


# In[25]:


print(grouped_person.groups.keys())
len(grouped_person.groups.keys())


# In[26]:


grouped_person.size()


# In[ ]:




