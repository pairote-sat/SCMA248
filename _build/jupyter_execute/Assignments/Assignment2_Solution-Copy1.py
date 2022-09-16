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
# Over a thousand local government units are covered by the fund, which charges about \$25 million in annual premiums and provides insurance coverage of about \$75 billion.

# **Example 1** Import the claim dataset namely ClaimsExperienceData.csv from my Github repository. Then write Python commands to answer the following questions.

# In[1]:


import pandas as pd

claims = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/ClaimsExperienceData.csv')


# 1. How many claims observations are there in this dataset?

# In[2]:


claims.shape[0]


# 2. How many variables (features) are there in this dataset? List (print out) all the features. 

# In[3]:


claims.shape[1]
claims.columns


# 
# ## Description of Rating Variables
# 
# One of the important tasks of insurance analysts is to develop models to represent and manage the two outcome variables, **frequency** and **severity**. 
# 
# However, when actuaries and other financial analysts use those models, they do so in the context of external variables. 
# 
# In general statistical terminology, one might call these explanatory or predictor variables.
# 
# Because of our insurance focus, we call them **rating variables** as they are useful in setting insurance rates and premiums.
# 
# The following table describes the rating variables considered.
# 
# These are variables that you think might naturally be related to claims outcomes.
# 
# <!-- To handle the skewness, we henceforth focus on logarithmic transformations of coverage and deductibles. -->
# 
# <!-- For our immediate purposes, the coverage is our first rating variable. Other things being equal, we would expect that policyholders with larger coverage have larger claims. We will make this vague idea much more precise as we proceed, and also justify this expectation with data. -->
# 
# **Variable**  | **Description**
# ----- | -------------
# EntityType    | Categorical variable that is one of six types: (Village, City, County, Misc, School, or Town) 
# LnCoverage    | Total building and content coverage, in logarithmic millions of dollars
# LnDeduct      | Deductible, in logarithmic dollars
# AlarmCredit   | Categorical variable that is one of four types: (0, 5, 10, or 15) for automatic smoke alarms in main rooms
# NoClaimCredit | Binary variable to indicate no claims in the past two years
# Fire5         | Binary variable to indicate the fire class is below 5 (The range of fire class is 0 to 10)  

# **In what follows, for illustrate, we will consider claims data in year 2010.**

# 3. How many policies are there in 2010? 
# 
# Name the answer with the variable name **num_policies**. 
# 
# Hint: one may use `.value_counts` method that return a Series containing counts of unique values. Alternatively, you want to count False and True separately you can use `pd.Series.sum()` + `~`.

# In[4]:


temp = claims['Year']  == 2010
temp.value_counts()


# In[5]:


num_policies = temp.sum()


# In[6]:


(~temp).sum()


# 4. How many claims are there in 2010? Assign the result to the variable **num_claims**.

# In[7]:


claims2010 = claims[temp]


# In[8]:


claims2010.columns


# In[9]:


claims2010.sum()


# In[10]:


num_claims = claims2010['Freq'].sum()
print(num_claims)


# 5. Which policy number has the maximum number of claims and what is this claims number?

# In[11]:


claims2010.sort_values('Freq', ascending = False).head(2)


# # Hard cording
# 
# claims2010.loc[1406,'Freq'] 

# With `.idxmax()`, we can return the index at which maximum weight value is present.
# 
# See https://www.geeksforgeeks.org/get-the-index-of-maximum-value-in-dataframe-column/.

# In[12]:


print(claims2010['Freq'].idxmax())

ind_freq_max = claims2010['Freq'].idxmax()

max_claims = claims2010.loc[ind_freq_max,'Freq'] 


# 6. Calculate the proportion of policyholders who did not have any claims (use the name **num_policies_no_claims** for your output).

# In[13]:


# Using value_count() and .sort_index to obtain the number of 
# policies by claim numbers.

(claims2010['Freq'].value_counts()).sort_index()

num_policies_no_claims = (claims2010['Freq'].value_counts()).sort_index()[0]


# In[14]:


# Calculate the proportion of policyholders who did not have any claims.

round(num_policies_no_claims / num_policies,4)


# In[15]:


(claims2010['Freq'].value_counts())[0]/claims2010['Freq'].sum()


# 7. Calculate the proportion of policyholders who had only one claim.

# In[16]:


num_policies_one_claims = (claims2010['Freq'].value_counts()).sort_index()[1]


# In[17]:


round(num_policies_one_claims / num_policies,4)


# 8. Calculate the average number of claims for this sample. 

# In[18]:


num_claims/num_policies


# 9. The `describe()` method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame. 
# 
# Applying to year 2010, what do we get when we run the command claims.describe()?

# In[19]:


claims2010.describe()


# 10. A common method for determining the severity distribution is to look at the distribution of the sample of 1,377 claims. Another typical strategy is to look at the **distribution of average claims among policyholders who have made claims**.
# 
# In our 2010 sample, how many such policyholders who have made claims?

# In[20]:


num_policies - num_policies_no_claims


# 11. The average claim for the 209 policyholders who had only one claim is the same as the single claim they had. 
# 
# Write the command(s) to list the average claim of such 209 policyholders.

# In[21]:


selected_index = (claims2010['Freq'] == 1)

claims2010[selected_index][['Freq','y']]


# 12. Calculate the average claim of the policyholder with the maximum number of claims.
# 
# ind_freq_max = claims2010['Freq'].idxmax()
# 
# max_claims = claims2010.loc[ind_freq_max,'Freq'] 

# In[22]:


claims2010.loc[ind_freq_max,'y'] / claims2010.loc[ind_freq_max,'Freq'] 


# In[23]:


claims.describe()


# In[24]:


claims.mean()


# ## Part 2

# 1. Write Python code to generage a table that shows the 2010 claims frequency distribution. The table should contain the number of policies, the number of claims and the proportion (broken down by the number of claims).
# 
# Goal: the table should tell us how many poicyholders and the (percentage) proportion of policyholders who did not have any claims, only one claim and so on.
# 
# 1.1. How many policyholders in the 2010 claims data have 9 or more claims?
# 
# 1.2. What is the percentage proportion of policyholders having exactly 3 claims?

# In[25]:


import pandas as pd

#claims = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/ClaimsExperienceData.csv')
url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/ClaimsExperienceData.csv'
claims = pd.read_csv(url)


# To obtain the number of policies and number of claims categorized by claim frequency (and by year), we first group `claims` dataset by `Year` and `Freq` and apply `agg` (aggregate) function.

# In[3]:


claimsFreqDist = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count', 'Freq' : 'sum'})
claimsFreqDist.rename(columns={'PolicyNum':'NumPolicies', 'Freq':'NumClaims'}, inplace = True)
claimsFreqDist


# To determine the (percentage) proportion of policies for each claim frequency, we will first group the "Year" and "Freq".   To calculate the percentage **within each "Year" group**, the following command can be used groupby(level=0).apply(lambda x: 100*x/x.sum())
# 
# **Note:** Because the original dataframe becomes a multiple index dataframe after grouping, the level = 0 refers to the top level index, which in our case is 'Year'.
# 
# You can see the results below, which have already been sorted by percent of policies for each claim frequency by year.

# In[12]:


#Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())
Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())
Table1.rename(columns={'PolicyNum':'Percentage'}, inplace = True)

#Table1.loc[(2010,slice(None))]
Table1.loc[(slice(None),slice(None))].tail(30)


# The final is to merged the above two resulting tables using `pd.merge` function.

# In[13]:


claimsFreqDist = pd.merge(Table1,claimsFreqDist, how = 'left', on = ['Year','Freq'])

#claimsFreqDist

claimsFreqDist.loc[(2010,slice(None))].sort_index(axis=1)


# In[318]:


#claimsFreqDist.loc[(2010,slice(None))].sort_index(axis=1)

claimsFreqDist.index
claimsFreqDist.loc[(slice(None),[1,2,5]),['Percentage','NumClaims'] ].head()


# 1.1. How many policyholders in the 2010 claims data have 9 or more claims?

# In[58]:


# see for more detail: https://stackoverflow.com/questions/50608749/slicing-a-multiindex-dataframe-with-a-condition-based-on-the-index

#l0 and l1 are Year and Freq levels, respectively.
l0 = claimsFreqDist.index.get_level_values(0)
l1 = claimsFreqDist.index.get_level_values(1)
cond = (l0 == 2010) & (l1 >= 9)

claimsFreqDist.loc[cond,'NumPolicies'].sum()

ans1_1 = claimsFreqDist.loc[cond,'NumPolicies'].sum()

print('Ans: The number of policyholders in the 2010 claims data having 9 or more claims is'
      , ans1_1)


# In[309]:


# https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

claimsFreqDist.query('Year == 2010 & Freq >= 9')['NumPolicies'].sum()


# 1.2. What is the percentage proportion of policyholders having exactly 3 claims?

# In[59]:


claimsFreqDist.query('Year == 2010 & Freq == 3')['Percentage']

ans1_2 = (claimsFreqDist.query('Year == 2010 & Freq == 3')['Percentage']).values[0]


print('Ans: The percentage proportion of policyholders having exactly 3 claims:', ans1_2)


# In[43]:


claims2010 = claims[claims['Year']  == 2010]


# In[44]:


claims2010.head()


# In[49]:


claims.columns

claims.index.names


# This outputs a “FrozenList”, which is just a Pandas specific construct used to show the index label(s) of a DataFrame. Here, we see the value is “None”, as this is the default value of a DataFrame’s index.

# To create a MultiIndex with our original DataFrame, all we need to do is pass a list of columns into the .set_index() Pandas function like this:

# In[52]:


multi = claims.set_index(['Year','Freq'])

multi.tail(50)


# In[6]:


multi.index.names


# In[7]:


multi.index.values


# In[9]:


multi = claims.set_index(['Year','Freq']).sort_index()

multi.head()


# In[11]:


multi.reset_index()


# In[22]:


multi.loc[(2010,0),['y','Deduct']]


# In[25]:


multi.loc[([2008,2010],[0,1]),['y','Deduct']]


# In[5]:


out = claims.groupby(by = ['Year','Freq']).count()
out.head()


# In[31]:


out.index


# In[63]:


out.loc[[2006,], 'y']


# In[68]:


out.loc[[2010,], 'y']


# In[69]:


out.index.names


# Start here!!!

# Again, we pass a tuple in with our desired index values, but instead of adding values for “freq”, we pass `slice(None)`. This is the default `slice` command in Pandas to select all the contents of the MultiIndex level.

# In[52]:


out2010 = out.loc[(2010,slice(None)),:]


# In[45]:


out2010.columns


# In[46]:


100*out2010['PolicyNum']/(out2010['PolicyNum'].sum())


# In[47]:


out2010.shape


# In[28]:


out2010.index


# In[42]:


(out2010.loc[(2010,slice(None)),:])['Proportion']


# In[25]:


(100*out2010['PolicyNum']/(out2010['PolicyNum'].sum())).shape


# In[48]:


out2010['Proportion'] = 100*out2010['PolicyNum']/(out2010['PolicyNum'].sum())


# In[51]:


out2010.loc[:,['PolicyNum','Proportion']]


# In[53]:


out2010 = out.loc[(2010,slice(None)),:]

out2010.index


# In[65]:


# This works pretty simply, but the resulting DataFrames no longer have the multi-index. Also .xs() is not the most powerful way to subset a DataFrame.
# https://www.somebits.com/~nelson/pandas-multiindex-slice-demo.html


out2010 = (out.xs(2010))

print(out2010.index)
out2010['Proportion'] = 100*out2010['PolicyNum']/(out2010['PolicyNum'].sum())

out2010.loc[:,['PolicyNum','Proportion']]


# ## Next attempt

# In[106]:


table1 = claims.loc[:,['Year','Freq','PolicyNum']].groupby(by = ['Year','Freq']).count()
#table1['PolicyNum']

#print(table1.loc[(2006,),:])

#print(table1.head())

table1.groupby('Year').sum()

table1.loc[(2006,slice(None))]


# ## Pivot

# In[26]:


print(claims.columns)
func = lambda x: 100*x.count()/1110
claims.pivot_table(index = 'Freq', columns = 'Year', values = ['PolicyNum','PolicyNum'], aggfunc = ['count','sum'])


# ## Groupby

# In[49]:


print(claims.groupby(['Year','Freq'])['PolicyNum'].count())

print(claims.groupby(['Year','Freq'])['PolicyNum'].count().groupby(level='Year').sum())

100*(claims.groupby(['Year','Freq'])['PolicyNum'].count().loc[(2010,slice(None))]/claims.groupby(['Year','Freq'])['PolicyNum'].count().groupby(level='Year').sum()[2010])


# In[60]:


# not working

output = claims.groupby(['Year','Freq'])['PolicyNum'].count().rename('Percentage').transform(lambda x: x/x.sum())
output


# ## Last Attempt:

# To determine the (percentage) proportion of policies for each claim frequency, we will first group the "Year" and "Freq".   To calculate the percentage **within each "Year" group**, the following command can be used groupby(level=0).apply(lambda x: 100*x/x.sum())
# 
# **Note:** Because the original dataframe becomes a multiple index dataframe after grouping, the level = 0 refers to the top level index, which in our case is 'Year'.
# 
# You can see the results below, which have already been sorted by percent of sales contribution for each sales person.

# In[88]:


#Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())

Table1 = claims.groupby(['Year','Freq']).agg({'PolicyNum' : 'count'}).groupby(level='Year').apply(lambda x: 100*x/x.sum())

Table1.rename(columns={'PolicyNum':'Percentage'})

Table1.loc[(2010,slice(None))]


# 2. From those 403 policyholders who made at least one claim, create a table that provides information about the distribution of average claim amounts in year 2010.
# 
# 2.1. What is the mean of the average claim amounts?
# 
# 2.2. What is the third quartile of the average claim amounts?

# First, we add the column, namely `ClaimsAvg` representing the average cost per claim for each observation. The average cost per claim (or claim average) amount is calculated by dividing the number of claims  by the total claim amount.

# In[53]:


claims['ClaimsAvg'] = claims['y']/claims['Freq']
claims['ClaimsAvg'] = claims['ClaimsAvg'].fillna(0)
claims['ClaimsAvg']


# 
# 

# The information about the distribution of average claim amounts in year 2010 is given in the table below including count, mean, std, min, 25, 50, 75% percentiles and max.

# In[423]:


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

# In[ ]:




