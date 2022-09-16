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
# Over a thousand local government units are covered by the fund, which charges about $25 million in annual premiums and provides insurance coverage of about $75 billion.

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


# In[ ]:





# In[ ]:





# In[23]:


claims.describe()


# In[24]:


claims.mean()


# In[ ]:




