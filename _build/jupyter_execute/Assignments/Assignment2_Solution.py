#!/usr/bin/env python
# coding: utf-8

# ## Exercise for Chapter 3
# 
# This exercises is design to assist you how to use the pandas package to import, preprocess data and perform basic statistical analysis. Later we should see how data generating events can produce data of interest to insurance analysts.
# 
# We will look at the Local Government Property Insurance Fund in this chapter. The fund insures property owned by municipal governments, such as schools and libraries.
# 
# * government buildings,
# 
# * educational institutions,
# 
# * public libraries, and
# 
# * motor vehicles.
# 
# Over a thousand local government units are covered by the fund, which charges about \\$25 million in annual premiums and provides insurance coverage of about \\$75 billion.

# ## Part 2

# 1. Write Python code to generage a table that shows the 2010 claims frequency distribution. The table should contain the number of policies, the number of claims and the proportion (broken down by the number of claims).
# 
# Goal: the table should tell us how many poicyholders and the (percentage) proportion of policyholders who did not have any claims, only one claim and so on.
# 
# 1.1. How many policyholders in the 2010 claims data have 9 or more claims?
# 
# 1.2. What is the percentage proportion of policyholders having exactly 3 claims?

# In[1]:


import pandas as pd

#claims = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/ClaimsExperienceData.csv')
url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/ClaimsExperienceData.csv'
claims = pd.read_csv(url)


# To obtain the number of policies and number of claims categorized by claim frequency (and by year), we first group `claims` dataset by `Year` and `Freq` and apply `agg` (aggregate) function.

# In[20]:


claimsFreqDist = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count', 'Freq' : 'sum'})
claimsFreqDist.rename(columns={'PolicyNum':'NumPolicies', 'Freq':'NumClaims'}, inplace = True)
#claimsFreqDist


# To determine the (percentage) proportion of policies for each claim frequency, we will first group the "Year" and "Freq".   To calculate the percentage **within each "Year" group**, the following command can be used groupby(level=0).apply(lambda x: 100*x/x.sum())
# 
# **Note:** Because the original dataframe becomes a multiple index dataframe after grouping, the level = 0 refers to the top level index, which in our case is 'Year'.
# 
# You can see the results below, which have already been sorted by percent of policies for each claim frequency by year.

# In[21]:


#Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())
Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())
Table1.rename(columns={'PolicyNum':'Percentage'}, inplace = True)

#Table1.loc[(2010,slice(None))]
#Table1.loc[(slice(None),slice(None))].tail()


# The final is to merged the above two resulting tables using `pd.merge` function.

# In[22]:


claimsFreqDist = pd.merge(Table1,claimsFreqDist, how = 'left', on = ['Year','Freq'])

#claimsFreqDist

claimsFreqDist.loc[(2010,slice(None))].sort_index(axis=1)


# In[23]:


#claimsFreqDist.loc[(2010,slice(None))].sort_index(axis=1)

#claimsFreqDist.index
#claimsFreqDist.loc[(slice(None),[1,2,5]),['Percentage','NumClaims'] ].head()


# 1.1. How many policyholders in the 2010 claims data have 9 or more claims?

# In[24]:


# see for more detail: https://stackoverflow.com/questions/50608749/slicing-a-multiindex-dataframe-with-a-condition-based-on-the-index

#l0 and l1 are Year and Freq levels, respectively.
l0 = claimsFreqDist.index.get_level_values(0)
l1 = claimsFreqDist.index.get_level_values(1)
cond = (l0 == 2010) & (l1 >= 9)

claimsFreqDist.loc[cond,'NumPolicies'].sum()

ans1_1 = claimsFreqDist.loc[cond,'NumPolicies'].sum()

print('Ans: The number of policyholders in the 2010 claims data having 9 or more claims is'
      , ans1_1)


# In[25]:


# https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

claimsFreqDist.query('Year == 2010 & Freq >= 9')['NumPolicies'].sum()


# 1.2. What is the percentage proportion of policyholders having exactly 3 claims?

# In[26]:


claimsFreqDist.query('Year == 2010 & Freq == 3')['Percentage']

ans1_2 = (claimsFreqDist.query('Year == 2010 & Freq == 3')['Percentage']).values[0]


print('Ans: The percentage proportion of policyholders having exactly 3 claims:', ans1_2)


# 2. From those 403 policyholders who made at least one claim, create a table that provides information about the distribution of average claim amounts in year 2010.
# 
# 2.1. What is the mean of the average claim amounts?
# 
# 2.2. What is the third quartile of the average claim amounts?

# First, we add the column, namely `ClaimsAvg` representing the average cost per claim for each observation. The average cost per claim (or claim average) amount is calculated by dividing the number of claims  by the total claim amount.

# In[8]:


claims['ClaimsAvg'] = claims['y']/claims['Freq']
claims['ClaimsAvg'] = claims['ClaimsAvg'].fillna(0)
#claims['ClaimsAvg']


# 
# 

# The information about the distribution of average claim amounts in year 2010 is given in the table below including count, mean, std, min, 25, 50, 75% percentiles and max.

# In[9]:


#claimsFreqDist.query('Year == 2010 & Freq >= 1')

#claims[['Year','y','Freq']][0:10]
#claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['y'].describe()
claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['ClaimsAvg'].describe()


# 2.1. What is the mean of the average claim amounts?

# In[57]:


# the mean of the average claim amounts

(claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['ClaimsAvg']).mean()

ans2_1 = (claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['ClaimsAvg']).mean()

print('Ans: The mean of the average claim amounts: ', ans2_1)


# 2.2 What is the third quartile of the average claim amounts?

# In[60]:


# the third quartile of the average claim amounts
(claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['ClaimsAvg']).quantile(q = 0.75)

ans2_2 = (claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['ClaimsAvg']).quantile(q = 0.75)

print('Ans: the third quartile of the average claim amounts:', ans2_2)


# In[61]:


# the mean of the average claim amounts

# (claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['yAvg']).mean()


# In[62]:


# the third quartile of the average claim amounts
# (claims[(claims['Year']==2010) &  (claims['Freq'] >= 1)]['yAvg']).quantile(q = 0.75)


# 3. Consider the claims data over the 5 years between 2006-2010 inclusive. Create a table that show the average claim varies over time, average frequency, average coverage and the number of policyholders. 
# 
# 3.1 What can you say about the number of policyholders over this period?
# 
# 3.2 How does the average coverage change over this period?

# In[425]:


claims.columns


# In[434]:


claims.groupby('Year').agg({'Freq':'mean', 'ClaimsAvg':'mean', 'BCcov':'mean', 'PolicyNum':'count' })


# In[63]:


# claims.groupby('Year').agg({'Freq':'mean', 'yAvg':'mean', 'BCcov':'mean', 'PolicyNum':'count' })


# ## Conclusion
# 
# 1. The table shows that the average claim varies over time, especially with the high 2010 value (that we saw was due to a single large claim).
# 
# 2. The total number of policyholders is steadily declining and, conversely, the coverage is steadily increasing (**Answers of Questions 3.1 and 3.2**).
# 
# 3. The coverage variable is the amount of coverage of the property and contents. Roughly, you can think of it as the maximum possible payout of the insurer.

# Ans: 
# 
# 3.1 The total number of policyholders is steadily declining and, conversely, 
# 
# 3.2 the coverage is steadily increasing.

# In[27]:


print('Ans 1.1:', ans1_1 ,'\n', 'Ans 1.2:', ans1_2 ,'\n')


# In[ ]:




