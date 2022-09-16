#!/usr/bin/env python
# coding: utf-8

# # Midterm Examination: Part 1 
# 
# Due date: 15 March 2022 before 10 am.
# 
# Points: 15 for Part 1 (from 21 questions)
# 
# Do not alter this file.
# 
# Duplicate this file and move it into the Personal folder. Rename the file as Midterm_Part1_id (id is your student ID number, e.g. Midterm_Part1_6305001).
# 
# Write Python commands to answer each of the questions. For each question that requires numerical values (not list or dataframe), you also need to assign the variable e.g. ans1 to store the numerical answer for question 1. If there is more than 1 answer required, you must create more variables e.g. ans1_1, ans1_2 to store the values of the answers.
# 
# When you want to submit your file, you simply share access with me using my email pairote.sat@mahidol.edu and my TA p.pooy.pui.i@gmail.com. Do not move your file into the DS@MathMahidol team.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

pd.options.mode.chained_assignment = None


# In[3]:


from plotnine import *


# In[4]:


id = 888


# ## Population Dataset: 2020 Census Demographic Data by County.
# 
# We begin with the first dataset, which we obtained from 
# 
# https://www.dataemporium.com/dataset/254/?gclid=CjwKCAiAg6yRBhBNEiwAeVyL0Jl9xZg-nt9evBLB04fAZPc-TPTEmrW9kMfoIqMhBJvHjXQ-GV5fPBoChYIQAvD_BwE
# 
# This dataset was created from the 2020 Census. It contains one row per county with the total population and a breakdown by race.

# In[5]:


path = '/Users/Kaemyuijang/SCMA248/Data/US_population.csv'

#df = pd.read_csv(path, parse_dates=True, index_col = 'date')

population = pd.read_csv(path)


# #### Detailed description
# 
# This table breaks down the total population and population by race for each county in the United States, based on the 2020 Census.
# 
# **Some Terminology**.
# 
# * A city is created by any population that has its own system of government and a semblance of a legal system. Cities are located within a county, within a state. 
# 
# * A county is a geographic unit created for political purposes within a state.
# 
# Read more : Difference between city and county 
# http://www.differencebetween.net/miscellaneous/difference-between-city-and-county/#ixzz7NKcARPB5
# 
# **This table only includes numbers for people who checked off a single race in the census**. This is the majority of people. You can see the number of people who belong to two or more races by subtracting POP_ONE_RACE from TOTAL_POPULATION. 
# 
# Before proceeding to the next step, make sure you understand the difference between POP_ONE_RACE from TOTAL_POPULATION.

# In[6]:


population.columns= population.columns.str.lower()

population.head()


# Q: Verify that the sum of counts in the columns ['white','black','amarican_indian','asian','hawaiian','other'] is equal to the count in the column 'pop_one_race'.

# In[7]:


population["sum"] = population[['white','black','amarican_indian','asian','hawaiian','other']].sum(axis=1)
population
ans1 = population['pop_one_race'].equals(population['sum'])
ans1


# Q: How many counties are included in this dataset?

# In[8]:


population.county.size


# Q: How many unique county names are there?

# In[9]:


population.county.unique().size


# Q: What can be concluded from the difference between the number of counties and the number of unique county names?

# In[10]:


population.shape


# Q: Create a table with the number of counties in each U.S. state.

# In[11]:


population.groupby('state_abbr').size()


# Q: The following Python command uses `np.random.seed(id)` to set the seed number of your student ID. Complete the following command to create the variable 'state_given', which stores a randomly selected US state.

# In[12]:


np.random.seed(id)
states_given = np.random.choice(population.state_abbr.unique(), 1).tolist()
print(states_given)


# Q: List all counties in the randomly selected US state defined by `states_given`.

# In[13]:


population.query('state_abbr == @states_given')


# In[14]:


# How to use loop variable inside pandas df.query()
# https://stackoverflow.com/questions/64828539/how-to-use-loop-variable-inside-pandas-df-query

states_list = ["TX","WY"]

#print(population.query('states_list == "AL"')[['county']])

for st in states_list:
    print(st)
    #print('states_list ==' + str(st))
       
    print(population.query('state_abbr ==@st')[['county']])


# In[15]:


population.state_abbr.unique()


# Q: Write Python code to select two random US states and include them in a list named **states_list**. Do not forget to use np.random.seed to set the random value to your ID.

# In[16]:


np.random.seed(id)
states_list = np.random.choice(population.state_abbr.unique(), 2, replace = True).tolist()


# Q: List all counties for each state in the **states_list**.

# In[17]:


for st in states_list:
    print(st)
    #print('states_list ==' + str(st))
       
    #print(population.query('state_abbr ==@st')[['county']])
    print(population.query('state_abbr ==@st'))


# In[ ]:





# #### Population by US states

# Q: Write Python code to create a table including the population for each U.S. state. The table should include total population, pop_one_race, white, black, etc. for each state.

# In[18]:


population_state = population.groupby('state_abbr').sum()


# In[19]:


population_state.head()


# Q: From the table of population by race for each US state created above, write Python code to calculate the (row) percentages (for each US state) of the following variables 
# **'white','black','amarican_indian','asian','hawaiian','other'**.

# In[20]:


races_list = ['white','black','amarican_indian','asian','hawaiian','other']

population_state[races_list] = population_state[races_list].apply(lambda x: x/x.sum(), axis=1)


# In[21]:


population_state.head()


# In[22]:


population_state['state_abbr'] = population_state.index


# In[23]:


population_state


# Q: List the first five states with the highest percentage of white Americans.

# In[24]:


population_state.sort_values(by = ['white'], ascending = False).head(5)


# Q: List the first five states with the highest percentage of black Americans.

# In[25]:


population_state.sort_values(by = ['black'], ascending = False).head(5)


# #### Visualization of U.S. population by race and U.S. states

# In[26]:


popultion_melted = pd.melt(population_state[races_list+['state_abbr']], id_vars=['state_abbr'], value_vars=races_list).rename(columns={"value":"proportion","variable":"race"})

print(popultion_melted)


# In[27]:


(
ggplot(popultion_melted.query('state_abbr in @states_list')) 
    + aes(x = 'state_abbr', y = 'proportion',fill = 'race')
    + geom_col(position = "fill")
    + theme(axis_text_x=element_text(rotation=90, hjust=1)) 
    #+ facet_wrap('state_abbr')
)


# Q: Write Python to graph the US population by race broken down by state.

# In[28]:


(
ggplot(popultion_melted) 
    + aes(x = 'state_abbr', y = 'proportion',fill = 'race')
    + geom_col(position = "fill")
    + theme(axis_text_x=element_text(rotation=90, hjust=4)) 
)


# In[ ]:





# ## US Covid-19 Dataset

# The second dataset can be downloaded from 
# 
# https://www.kaggle.com/fireballbyedimyrnmom/us-counties-covid-19-dataset.
# 
# Each data row contains data on cumulative coronavirus cases and deaths.
# 
# The specific data here, are the data **PER US COUNTY**.
# 
# We will work with this Covid-19 dataset by preprocessing data, performing statistical data analysis and presenting the results.

# In[29]:


path = '/Users/Kaemyuijang/SCMA248/Data/us-counties.csv'

#df = pd.read_csv(path, parse_dates=True, index_col = 'date')

df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')


# #### Data Cleaning and Preparation: Handling Missing Values

# In[30]:


df.shape


# Q: List all variables (or columns) in this Covid-19 dataset with missing data.

# In[31]:


# Find columns with missing data
# https://moonbooks.org/Articles/How-to-filter-missing-data-NAN-or-NULL-values-in-a-pandas-DataFrame-/

df.isnull().any()


# In[32]:


# Get a list of columns with missing data

df.columns[df.isnull().any()]


# Q: Calculate the number of missing data for each variable (i.e. in each column).

# In[33]:


# Get the number of missing data per column

df.isnull().sum()


# #### States with missing fips.

# In[34]:


(df[df.fips.isnull()]).state.unique()


# #### Filtering out Missing Data

# ##### Droping out row with missing fips

# Here is the list of observations with missing fips.

# In[35]:


(df[df.fips.isnull()])


# Q: Write Python code to drop rows only when the column 'fips' has NaN in it. How many rows are there in the resutling DataFrame.

# In[36]:


print(df.dropna(subset=['fips']))

df.dropna(subset=['fips']).shape


# Now we can varify that we have successfully dropped rows with NaN values only in the fips column.

# In[37]:


(df.dropna(subset=['fips'])).fips.isnull().sum()


# Q: How many NaN values remain in the **Deaths** column in the resutling DataFrame after deleting only rows where the  'fips' column contains NaN?

# In[38]:


(df.dropna(subset=['fips'])).deaths.isnull().sum()


# In[39]:


df.shape


# #### List of obeservations with NA in both **fips** and **deaths** column.
# 
# We want to select only those rows from this DataFrame, where columns **fips** and **deaths** have NaN values i.e.
# 
# See Select Dataframe Rows with NaN in multiple columns
# 
# https://thispointer.com/pandas-select-rows-with-nan-in-column/
# 

# In[40]:


(df[df.fips.isnull() & df.deaths.isnull()])


# In[ ]:





# Q: Drop all rows with missing values in the deaths column

# In[41]:


df.dropna(inplace=True)


# In[42]:


df.shape


# In[ ]:





# In[43]:


population.head()


# In[ ]:





# ## Merging multiple DataFrames
# 
# The Covid-19 dataset contains only data on cumulative coronavirus cases and deaths. To perform our analysis, e.g., to compare infection rates among different U.S. states, we need to add more information on the population size of each county.
# 
# We will need to download another data frame that contains the list of state abbreviations. We will then combine multiple data frames by adding the 'total_population' to the Covid-19 dataset.
# 
# * List of State Abbreviations: 
# 
# https://worldpopulationreview.com/states/state-abbreviations

# In[44]:


path = '/Users/Kaemyuijang/SCMA248/Data/us_state_abbreviations.csv'

abbr = pd.read_csv(path)


# In[45]:


abbr.columns= abbr.columns.str.lower()

#abbr.rename(columns={"State":"state","Abbrev":"abbrev","Code":"code"}, inplace = True)

abbr.head()


# In[46]:


population.head()


# In[47]:


df.head()


# Q: Using the population and us_state_abbreviations datasets, write Python code to add the 'total_population' to the Covid-19 dataset.

# In[48]:


df_index = df.index


# In[49]:


df = pd.merge(df,abbr[['state','code']], how='left', on='state')
df.index = df_index


# In[50]:


df.head()


# In[51]:


population.head()


# #### Removing 'County' from the county names of the **population** data set.

# In[52]:


# https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
population['county'] = population['county'].map(lambda x: x.rstrip(' County'))


# In[53]:


population.county.unique().size


# In[54]:


# The number of unique county names in population
population.county.unique().size


# In[55]:


# The number of unique county names in Covid-19 data
df.county.unique().size


# In[56]:


df.head()


# In[57]:


# df.drop(columns=['total_population_x','total_population_y'], inplace = True)


# In[58]:


df.columns


# In[59]:


print(df.shape)
df.head()


# In[60]:


population.head()


# In[61]:


# df = pd.merge(df,population[['county','code','total_population']], how='inner',on=['county','code'])


# In[62]:


# df = pd.merge(df,population[['code','county','total_population']], how='left',left_on=['county','code'], right_on=['county','code'])


# In[63]:


# df = pd.merge(df,population[['county','total_population']], how='left', on='county')


# #### Merging two DataFrames
# 
# To add total population, rename the column **state_abbr** of the DataFrame population to **code** for merging 

# In[64]:


#abbr.rename(columns={"State":"state","Abbrev":"abbrev","Code":"code"}, inplace = True)

population.rename(columns={"state_abbr":"code"}, inplace = True)


# In[65]:


print(population.head())

print(df[['code','county']].head())


# In[66]:


print(df.columns)
df.shape


# In[67]:


print(population.columns)
population.shape


# In[68]:


len(population.query('code == "CO"').county.unique())


# In[69]:


len(df.query('code == "CO"').county.unique())


# In[70]:


df.shape


# In[71]:


df['date'] = df.index


# In[72]:


df.head()


# In[73]:


temp = pd.merge(df,population[['code','county','total_population']], how='left', on=['county','code'])


# In[74]:


temp.shape


# In[75]:


temp.dropna(inplace = True)


# In[76]:


temp.shape


# In[77]:


# pd.merge(df,population[['code','county','total_population']], how='left', on=['county','code'])

df_merged = pd.merge(df,population[['code','county','total_population']], how='inner', on=['county','code'])
#df.index = df_index


# In[78]:


print(df_merged.head())

df_merged.isnull().sum()


# In[79]:


df_merged.shape


# In[80]:


df_merged.columns


# In[81]:


list(df_merged.columns.values)


# In[82]:


df_merged = df_merged[[ 'date','county', 'code','state', 'fips', 'cases', 'deaths','total_population']]


# In[83]:


df_merged


# In[84]:


len(df_merged.county.unique())


# In[85]:


df_merged.to_csv('/Users/Kaemyuijang/SCMA248/Data/US_covid19_part2.csv')


# In[86]:


path = '/Users/Kaemyuijang/SCMA248/Data/US_covid19_part2.csv'

final = pd.read_csv(path, parse_dates=True, index_col = 'date')


# In[87]:


final.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# #### Working on the merged DataFrame

# In[88]:


df = df_merged


# In[89]:


df.set_index('date')


# Q: Write Python to check for NaN values after adding 'total_population' to the Covid-19 dataset.

# In[90]:


df.isnull().sum()


# In[91]:


df.head()


# In[92]:


len(df.query('code == "CA"').county.unique())


# In[ ]:





# In[ ]:





# In[ ]:





# In[93]:


df.info()


# In[94]:


df.index


# In[95]:


df_Snohomish = df.query('county == "Snohomish"')


df_Snohomish['lag'] = df_Snohomish.cases.shift(1).fillna(0)
df_Snohomish['daily_cases'] = df_Snohomish.cases - df_Snohomish.lag

(
    ggplot(df_Snohomish) 
    + aes(x = df_Snohomish.index, y = 'cases') 
    + geom_line() 
    + theme(axis_text_x=element_text(rotation=90, hjust=1)) 
    + labs(x = "Cases", y = "Total Coronavirus Cases", title ="Total Coronavirus Cases in Snohomish")
)

#print(df_Snohomish.tail(10))

#df_Snohomish.query('daily_cases<0')


# In[96]:


(
    ggplot(df_Snohomish) 
    + aes(x = df_Snohomish.index, y = 'cases') 
    + geom_line() 
    + theme(axis_text_x=element_text(rotation=90, hjust=1)) 
    + labs(x = "Cases", y = "Total Coronavirus Cases", title ="Total Coronavirus Cases in Snohomish \n (logarithmic scale)")
    + scale_y_continuous(trans = "log10")
)


# In[97]:


(
    ggplot(df_Snohomish) 
    + aes(x = df_Snohomish.index, y = 'daily_cases') 
    + geom_line() 
    + theme(axis_text_x=element_text(rotation=90, hjust=1)) 
    + labs(x = "Cases", y = "New Coronavirus Daily Cases", title ="Daily New Cases in Snohomish")
)


# In[98]:


df_Snohomish.daily_cases[df_Snohomish.daily_cases < 0]


# In[99]:


df_Snohomish.loc['2022-02-01':'2022-02-18']


# In[100]:


df.query('state=="Washington"').head(20)


# In[ ]:





# In[101]:


path = '/Users/Kaemyuijang/SCMA248/Data/us-counties.csv'

#df = pd.read_csv(path, parse_dates=True, index_col = 'date')

df = pd.read_csv(path, parse_dates=['date'])


# In[102]:


df_Snohomish = df.query('county == "Snohomish"')

df_Snohomish['lag'] = df_Snohomish.cases.shift(1).fillna(0)
df_Snohomish['daily_cases'] = df_Snohomish.cases - df_Snohomish.lag


# In[103]:


pd_merge = pd.merge(df,df_Snohomish, how = 'left', on = ['date','county'])


# In[104]:


pd_merge.query('county == "Snohomish"')


# In[ ]:





# In[ ]:





# In[ ]:





# In[105]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[106]:


df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')

df = df.query('county in ["Snohomish","Cook"]')
#df = df.query('county in ["Snohomish"]')


# In[107]:


df.loc['2022-02-01':'2022-02-18']


# In[108]:


# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

pd.options.mode.chained_assignment = None


# In[109]:


df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    #print(temp)
    
    plt.plot(temp.index,temp.daily_cases)
    #print(temp.query('county == @ct'))
    #df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)


# In[110]:


# subplots_adjust ignores hspace, wspace #185
# https://github.com/has2k1/plotnine/issues/185

df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    
    rolling = temp.daily_cases.rolling(7, center=True)
    #print(rolling.mean().shape)
    #print(temp.shape)
    
    temp['rolling_mean'] = rolling.mean()
    #print(temp)
    
    df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)

(
    ggplot(df_new) + aes(x = df_new.index, y = 'daily_cases') + geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) + 
    #facet_grid('~fips', scales = "free_y")
    facet_wrap('fips', scales = "free_y") +
    theme(subplots_adjust={'wspace': 0.5,'hspace': 0.2})
)


# In[111]:


df_new.loc[df_new['fips'] == 13075.]['2022-02-01':'2022-02-18']


# In[112]:


#df_new13075 = df_new.loc[df_new['fips'] == 13075.]['2021-07-01':'2022-02-18']

df_new13075 = df_new.query('fips == 13075.')
#print(df_new13075.head(10))
(
    ggplot(df_new13075.query('fips == 13075.')) +
    geom_line(aes(x = df_new13075.index, y = 'daily_cases'), alpha = 0.5) +
    geom_line(aes(x = df_new13075.index, y = 'rolling_mean'), color='red',alpha=0.8) +
        theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[113]:


df_new17031 = df_new.query('fips == 17031.')

ggplot(df_new17031.query('fips == 17031.')) + aes(x = df_new17031.index, y = 'daily_cases') + geom_line()


# In[ ]:





# #### Query by state

# In[114]:


df = pd.read_csv(path, parse_dates=['date'], index_col = 'date')

df = df.query('state in ["Washington"]')
#df = df.query('county in ["Snohomish"]')

print(df.county.unique())


# In[115]:


df_new = pd.DataFrame()

for id in df['fips'].unique():
    print(id)
    temp = df.query('fips == @id')
    temp['lag'] = temp.cases.shift(1).fillna(0)
    temp['daily_cases'] = temp.cases - temp.lag
    temp.drop(columns=['lag'], inplace = True)
    
    rolling = temp.daily_cases.rolling(7, center=True)
    #print(rolling.mean().shape)
    #print(temp.shape)
    
    temp['rolling_mean'] = rolling.mean()
    #print(temp)
    
    df_new = pd.concat([df_new,temp])
    
    #print(temp)
    #df = pd.merge(df,temp, how = 'left', on =  ['date','county','state','fips','cases','deaths'])
    #df.drop(columns=['lag'], inplace = True)
    #print(df.columns)


# In[116]:


df_new.county.unique()


# In[117]:


fips_list = [53001., 53003.]

df_subset = df_new.query('fips in @fips_list')

(
    ggplot(df_subset) + aes(x = df_subset.index, y = 'daily_cases') + 
    geom_line() +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) + 
    facet_wrap('fips', nrow = 2, ncol = 1) +
    theme(subplots_adjust={'wspace': 0.5,'hspace': 0.2})
)


# In[118]:


#(
#    ggplot(df_new) + aes(x = df_new.index, y = 'daily_cases') + geom_line() +    
#    theme(axis_text_x=element_text(rotation=90, hjust=1)) + 
#    facet_wrap('fips', nrow = 15, ncol = 3) +
#    theme(subplots_adjust={'wspace': 0.5,'hspace': 0.2})
#)


# In[119]:


df_subset = df_new.query('fips in @fips_list')

(
    ggplot(df_subset) + geom_line(aes(x = df_subset.index, y = 'daily_cases'), alpha = 0.3) +    
    geom_line(aes(x = df_subset.index, y = 'rolling_mean'),color='red', alpha = 0.8) +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) +  
        facet_wrap('fips', nrow = 2, ncol = 1) +
    theme(subplots_adjust={'wspace': 0.5,'hspace': 0.2})
)


# In[120]:


fips_id = 53023.0

df_subset = df_new.query('fips==@fips_id')

(
    ggplot(df_subset) + geom_line(aes(x = df_subset.index, y = 'daily_cases'), alpha = 0.3) +    
    geom_line(aes(x = df_subset.index, y = 'rolling_mean'),color='red', alpha = 0.8) +    
    theme(axis_text_x=element_text(rotation=90, hjust=1)) 
)


# In[ ]:





# In[ ]:




