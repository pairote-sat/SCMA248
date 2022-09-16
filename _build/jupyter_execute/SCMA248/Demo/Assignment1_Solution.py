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
# Over a thousand local government units are covered by the fund, which charges about \$ 25 million in annual premiums and provides insurance coverage of about \$ 75 billion.

# **Example 1** Import the claim dataset namely ClaimsExperienceData.csv from my Github repository. Then write Python commands to answer the following questions.

# In[1]:


import pandas as pd

claims = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/ClaimsExperienceData.csv')


# 1. How many claims observations are there in this dataset?

# In[2]:


claims.shape[0]
my_ans1 = claims.shape[0]
print('Answer: There are', claims.shape[0],'claims in the dataset.')


# 2. How many variables (features) are there in this dataset? List (print out) all the features. 

# In[3]:


print(claims.columns)
claims.shape[1]
my_ans2 = claims.shape[1]
print('Answer: There are', claims.shape[1],'variables in the dataset.')


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
# 
# In addition, we may use Pandas Series.get() function get item from object for given key (DataFrame column, Panel slice, etc.).

# In[4]:


num_policies = (claims['Year']  == 2010).value_counts().get(True)
my_ans3 = num_policies
print('Answer: There are', num_policies,'policies in 2010.')


# In[5]:


# First Attempt

# temp = claims['Year']  == 2010
# temp.value_counts()
# num_policies = temp.sum()

#(~temp).sum()


# 4. How many claims are there in 2010? Assign the result to the variable **num_claims**.

# In[6]:


claims2010 = claims[ claims['Year']  == 2010]
num_claims = claims2010['Freq'].sum()
my_ans4 = num_claims
print('Answer: There are', num_claims,'claims in 2010.')


# 5. Which policy number has the maximum number of claims and what is this claims number?

# In[7]:


# claims2010.sort_values('Freq', ascending = False).head(2)
## Hard cording
# claims2010.loc[1406,'Freq']


# With `.idxmax()`, we can return the index at which maximum weight value is present.
# 
# See https://www.geeksforgeeks.org/get-the-index-of-maximum-value-in-dataframe-column/.

# In[8]:


ind_freq_max = claims2010['Freq'].idxmax()

max_policy_num = claims2010.loc[ind_freq_max,'PolicyNum'] 
max_claims = claims2010.loc[ind_freq_max,'Freq'] 

my_ans5_1 = max_policy_num
my_ans5_2 = max_claims

print('Ans: The policy number', max_policy_num, 'has the maximum number of claims of ', max_claims,'.')


# 6. Calculate the proportion of policyholders who did not have any claims (use the name **prop_policies_no_claims** for your output).

# In[9]:


# Using value_count() and .sort_index to obtain the number of 
# policies by claim numbers.

(claims2010['Freq'].value_counts()).sort_index()

num_policies_no_claims = (claims2010['Freq'].value_counts()).sort_index()[0]

# Calculate the proportion of policyholders who did not have any claims.

prop_policies_no_claims = round(num_policies_no_claims / num_policies,4)

my_ans6 = prop_policies_no_claims
print('Ans: The proportion of policyholders who did not have any claims is', prop_policies_no_claims)  


# 7. Calculate the proportion of policyholders who had only one claim.

# In[10]:


num_policies_one_claims = (claims2010['Freq'].value_counts()).sort_index()[1]

prop_policies_one_claims = round(num_policies_one_claims / num_policies,4)

my_ans7 = prop_policies_one_claims
print('Ans: The proportion of policyholders who did not have any claims is', prop_policies_one_claims)  


# 8. Calculate the average number of claims for this sample. 

# In[11]:


num_claims/num_policies

my_ans8 = num_claims/num_policies
print('Ans: The average number of claims for this sample is', round(num_claims/num_policies,4))


# 9. The `describe()` method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame. 
# 
# Applying to year 2010, what do we get when we run the command claims.describe()?

# In[12]:


claims2010.describe()


# 10. A common method for determining the severity distribution is to look at the distribution of the sample of 1,377 claims. Another typical strategy is to look at the **distribution of average claims among policyholders who have made claims**.
# 
# In our 2010 sample, how many such policyholders who have made claims?

# In[13]:


num_policies - num_policies_no_claims

my_ans10 = num_policies - num_policies_no_claims
print('Ans: There are',num_policies - num_policies_no_claims ,'policyholders who have at least made claims.')


# 11. The average claim for the 209 policyholders who had only one claim is the same as the single claim they had. 
# 
# Write the command(s) to list the average claim of such 209 policyholders.

# In[14]:


selected_index = (claims2010['Freq'] == 1)

claims2010[selected_index][['Freq','y','yAvg']]


# 12. Calculate the average claim of the policyholder with the maximum number of claims.

# In[15]:


ind_freq_max = claims2010['Freq'].idxmax()

max_yAvg = claims2010.loc[ind_freq_max,'y'] / claims2010.loc[ind_freq_max,'Freq'] 

my_ans12 = max_yAvg
print('Ans: the average claim of the policyholder with the maximum number of claims is ', round(max_yAvg,4),'.' )


# In[16]:


print('Q1:', my_ans1)
print('Q2:', my_ans2)
print('Q3:', my_ans3)
print('Q4:', my_ans4)
print('Q5_1:', my_ans5_1)
print('Q5_2:', my_ans5_2)
print('Q6:', my_ans6)
print('Q7:', my_ans7)
print('Q8:', my_ans8)
print('Q10:', my_ans10)
print('Q12:', my_ans12)


# ## Part 2

# 1. Create a table that shows the 2010 claims frequency distribution. The table should contain the number of policies, the number of claims and the proportion (broken down by the number of claims).
# 
# 1.1. How many policyholders in the 2010 claims data have 9 or more claims?
# 
# 1.2. What is the percentage proportion of policyholders having exactly 3 claims?
# 
# Goal: the table should tell us the (percentage) proportion of policyholders who did not have any claims, only one claim and so on. 

# 2. From those 403 policyholders who made at least one claim, create a table that provides information about the distribution of sample claims in year 2010.
# 
# 2.1. What is the mean of claims amounts?
# 
# 2.2. What is the third quartile of the claims amounts?

# 3. Consider the claims data over the 5 years between 2006-2010 inclusive. Create a table that show the average claim varies over time, average frequency, average coverage and the number of policyholders. 
# 
# 3.1 What can you say about the number of policyholders over this period?
# 
# 3.2 How does the average coverage change over this period?
