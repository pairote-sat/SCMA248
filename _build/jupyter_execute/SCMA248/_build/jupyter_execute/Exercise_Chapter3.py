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

# **Exercises** Import the claim dataset namely ClaimsExperienceData.csv from my Github repository. Then write Python commands to answer the following questions.
# 
# Follow the link below for more detailed attributes and methods of pandas:
# 
# https://www.w3resource.com/python-exercises/pandas/index.php

# 1. How many claims observations are there in this dataset?

# 2. How many variables (features) are there in this dataset? List (print out) all the features. 

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

# 4. How many claims are there in 2010? Assign the result to the variable **num_claims**.

# 5. Which policy number has the maximum number of claims and what is this claims number?

# With `.idxmax()`, we can return the index at which maximum weight value is present.
# 
# See https://www.geeksforgeeks.org/get-the-index-of-maximum-value-in-dataframe-column/.

# 6. Calculate the proportion of policyholders who did not have any claims (use the name **num_policies_no_claims** for your output).

# 7. Calculate the proportion of policyholders who had only one claim.

# 8. Calculate the average number of claims for this sample. 

# 9. The `describe()` method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame. 
# 
# Applying to year 2010, what do we get when we run the command claims.describe()?

# 10. A common method for determining the severity distribution is to look at the distribution of the sample of 1,377 claims. Another typical strategy is to look at the **distribution of average claims among policyholders who have made claims**.
# 
# In our 2010 sample, how many such policyholders who have made claims?

# 11. The average claim for the 209 policyholders who had only one claim is the same as the single claim they had. 
# 
# Write the command(s) to list the average claim of such 209 policyholders.

# 12. Calculate the average claim of the policyholder with the maximum number of claims.
