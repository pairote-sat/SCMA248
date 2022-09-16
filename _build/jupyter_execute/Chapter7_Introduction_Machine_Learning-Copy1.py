#!/usr/bin/env python
# coding: utf-8

# # Machine learning: Introduction
# 
# **Machine learning** is clearly one of the most powerful and significant technologies in the world today. And more importantly, we have yet to fully realize its potential. It will undoubtedly continue to make headlines for the foreseeable future.
# 
# Machine learning is **a technique for transforming data into knowledge**. In the last 50 years, there has been a data explosion. This vast amount of data is worthless until we analyze it and uncover the underlying patterns. 
# 
# Machine learning techniques are being used to **discover useful underlying patterns in complex data** that would otherwise be difficult to find. Hidden patterns and problem knowledge can be used to predict future events and make a variety of complex decisions.

# ## What Is Machine Learning?
# 
# Machine learning is the study of computer algorithms that can learn and develop on their own with experience and data. 
# 
# It is considered to be a component of **artificial intelligence**. 
# 
# Machine learning algorithms create a model based on **training data** to make predictions or decisions without having to be explicitly programmed to do so.

# ### Building models of data
# 
# It makes more sense to think of machine learning as a means of **building models of data**.
# 
# Machine learning is fundamentally about building mathematical models that facilitate the understanding of data. 
# 
# If we provide these models with **tunable parameters** that can be adapted to the observed data, we can call the program **"learning" from the data**. 
# 
# These models can be used to **predict and understand features of newly observed data** after fitting them to previously seen data.

# **Categories of Machine Learning**: image from ceralytics
# ![Categories of Machine Learning: from ceralytics](https://www.ceralytics.com/wp-content/uploads/2019/08/machine-learning.jpg)

# ## Applications of Machine learning
# 
# Machine learning tasks can be used for a variety of things. Here are some examples of traditional machine learning tasks: 
# 
# 1. **Recommendation systems** 
# 
# ![Netflix recommendation system: from medium](https://miro.medium.com/max/1400/1*QKQA8ylu1lCtOkJaa_gGaw.png)
# 
# 
# We come across a variety of online recommendation engines and methods. Many major platforms, such as Amazon, Netflix, and others, use these technologies. These recommendation engines use a machine learning system that takes into account user search results and preferences.
# 
# The algorithm uses this information to make similar recommendations the next time you open the platform.
# 
# You will receive notifications about new programs on Netflix. Netflix's algorithm checks the entire viewing history of its subscribers. It uses this information to suggest new series based on the preferences of its millions of active viewers. 
# 
# 
# 
# 
# The same recommendation engine can also be used to create ads. Take Amazon, for example. Let us say you go to Amazon to store or just search for something. Amazon's machine learning technology analyzes the user's search results and then generates ads as recommendations.
# 
# 
# 2. **Machine learning for Illness Prediction Healthcare use cases in healthcare**.
# 
# 
# ![Building Heart disease classifier using K-NN algorithm: image from https://cdn-images-1.medium.com/max/800/1*tGeiO5zee6exueRC8iBuaQ.jpeg](https://cdn-images-1.medium.com/max/800/1*tGeiO5zee6exueRC8iBuaQ.jpeg)
# 
# 
# Doctors can warn patients ahead of time if they can predict a disease. They can even tell if a disease is dangerous or not, which is quite remarkable. But even though using ML is not an easy task, it can be of great benefit.
# 
# In this case, the ML algorithm first looks for symptoms on the patient's body. It would use abnormal body functions as input, train the algorithm, and then make a prediction based on that. Since there are hundreds of diseases and twice as many symptoms, it may take some time to get the results.
# 
# 3. **Credit score - banking machine learning examples**.
# 
# It can be difficult to determine whether a bank customer is creditworthy. This is critical because whether or not the bank will grant you a loan depends on it.
# 
# Traditional credit card companies only check to see if the card is current and perform a history check. If the cardholder does not have a card history, the assessment becomes more difficult. For this, there are a number of machine learning algorithms that take into account the user's financial situation, previous credit repayments, debts and so on.
# 
# Due to a large number of defaulters, banks have already suffered significant financial losses. To limit these types of losses, we need an effective machine learning system that can prevent any of these scenarios from occurring. This would save banks a lot of money and allow them to provide more services to real consumers.

# ## Categories of Machine Learning
# 
# Machine learning can be divided into two forms at the most basic level: supervised learning and unsupervised learning.
# 
# #### Supervised learning
# Supervised learning involves determining how to model the relationship between measured data features and a label associated with the data; once this model is determined, it can be used to apply labels to new, unknown data. This is further divided into **classification** and **regression** tasks, where the **labels in classification are discrete categories** and the **labels in regression are continuous values**. 
# 
# ##### Classification problems (more examples below)
# 
# The output of a classification task is a discrete value. "Likes adding sugar to coffee" and "does not like adding sugar to coffee," for example, are discrete. There is no such thing as a middle ground. This is similar to teaching a child to recognize different types of animals, whether they are pets or not. 
# 
# The output (label) of a classification method is typically represented as an integer number such as 1, -1, or 0. These figures are solely symbolic in this situation. Mathematical operations should not be performed with them because this would be pointless. Consider this for a moment. What is the difference between "Likes adding sugar to coffee" and "does not like adding sugar to coffee"? Exactly. We won't be able to add them, therefore we won't.
# 
# ##### Regression problem (discussed in our last chatpter)
# 
# The outcome of a regression problem is a real number (a number with a decimal point). We could, for example, use the height and weight information to estimate someone's weight based on their height.
# 
# The data for a regression analysis will like the data in insurance data set. A **dependent variable** (or set of independent variables) and an **independent variable** (the thing we are trying to guess given our independent variables) are both present. 
# 
# We could state that height is the independent variable and weight is the dependent variable, for example.
# In addition, each row in the dataset is commonly referred to as an **example, observation, or data point**, but each column (without the **label/dependent variable**) is commonly referred to as a **predictor, independent variable, or feature**.
# 
# 
# ![Supervised learning: image from medium](https://miro.medium.com/max/1400/1*589X2eXJJkatGRG-z-s_oA.png)

# #### Unsupervised learning (more details in the next chapter)
# Unsupervised learning, sometimes known as "letting the dataset speak for itself," models the features of a dataset without reference to a label. Clustering and dimensionality reduction are among the tasks these models perform. 
# 
# **Clustering methods** find unique groups of data, while **dimensionality reduction** algorithms look for more concise representations.

# ### Supervised vs Unsupervised Learning: image from researchgate
# 
# ![Supervised vs Unsupervised Learning: image from researchgate](https://www.researchgate.net/publication/329533120/figure/fig1/AS:702267594399761@1544445050584/Supervised-learning-and-unsupervised-learning-Supervised-learning-uses-annotation_W640.jpg)

# ### Various classification, regression and clustering algorithms: image from scikit-learn
# 
# ![Various classification, regression and clustering algorithms: image from scikit-learn](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Scikit-learn_machine_learning_decision_tree.png/1024px-Scikit-learn_machine_learning_decision_tree.png)

# ## Machine Learning Basics with the K-Nearest Neighbors Algorithm
# 
# We will learn what **K-nearest neighbours (KNN)** is, how it works, and how to find the right k value. We will utilize the well-known Python library sklearn to demonstrate how to use KNN.
# 
# 
# ##### K-nearest neighbours can be summarized as follows:
# 
# * K- Nearest Neighbors is a **supervised machine learning** approach since the target variable is known,
# 
# * It is **non-parametric**, since no assumptions are made about the underlying data distribution pattern.
# 
# * It predicts the cluster into which the new point will fall based on feature similarity.
# 
# Both classification and regression prediction problems can be solved with KNN. However, since most analytical problems require making a decision, it is more commonly used in classification problems .
# 
# ### KNN algorithm's theory
# 
# The KNN algorithm's concept is one of the most straightforward of all the supervised machine learning algorithms. 
# 
# It simply calculates the distance between a new data point and all previous data points in the training set. 
# 
# Any form of distance can be used, such as 
# 
# * Euclidean or 
# 
# * Manhattan distances. 
# 
# The K-nearest data points are then chosen, where K can be any integer. Finally, the data point is assigned to the class that contains the majority of the K data points.
# 
# Note that the Manhattan distance, ${\displaystyle d_{1}}$, between two vectors ${\displaystyle \mathbf {p} ,\mathbf {q} }$  in an n-dimensional real vector space with fixed Cartesian coordinate system is defined as 
# 
# $$d_{1}(\mathbf {p} ,\mathbf {q} )=\|\mathbf {p} -\mathbf {q} \|_{1}=\sum _{i=1}^{n}|p_{i}-q_{i}|,$$
# where ${\displaystyle (\mathbf {p} ,\mathbf {q} )}$ are vectors
# 
# $${\displaystyle \mathbf {p} =(p_{1},p_{2},\dots ,p_{n}){\text{ and }}\mathbf {q} =(q_{1},q_{2},\dots ,q_{n})\,}.$$

# ### Example on KNN classifiers: image from Researchgate
# 
# ![Example on KNN classifiers: image from Researchgate](https://www.researchgate.net/profile/Mohammed-Badawy/publication/331424423/figure/fig1/AS:732056359297024@1551547245072/Example-on-KNN-classifier_W640.jpg)

# Our goal in this diagram is to identify a new data point with the symbol 'Pt' into one of three categories: "A," "B," or "C."
# 
# Assume that K is equal to 7. The KNN algorithm begins by computing the distance between point 'Pt' and all of the other points. The 7 closest points with the shortest distance to point 'Pt' are then found. This is depicted in the diagram below. Arrows have been used to denote the seven closest points.
# 
# The KNN algorithm's final step is to assign a new point to the class that contains the majority of the seven closest points. Three of the seven closest points belong to the class "B," while two of the seven belongs to the classes "A" and "C". Therefore the new data point will be classified as "B".
# 

# ### Dataset: https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from plotnine import *


# In[83]:


import warnings

warnings.filterwarnings( "ignore", module = "matplotlib\..*" )


# We will be using the fruit_data_with_colors dataset, avalable here at github page, https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt.
# 
# 
# The mass, height, and width of a variety of oranges, lemons, and apples are included in the file. The heights were taken along the fruit's core. The widths were measured perpendicular to the height at their widest point.
# 
# ##### Our goals
# To predict the appropriate fruit label, we'll use the mass, width, and height of the fruit as our feature points (target value).

# In[3]:


url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
    
df = pd.read_table(url)    


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.groupby('fruit_label').count()['fruit_name']


# In[7]:


plt.title('Fruit_label Count')
sns.countplot(df['fruit_name'])


# In[8]:


df.head()


# In[9]:


(
    ggplot(df) 
    + aes('fruit_name',fill = 'fruit_name') 
    + geom_bar()
    + scale_fill_manual(values=['red', 'olivedrab', 'gold', 'orange'])

)    


# In[10]:


# Check whether there are any missing values.

df.isnull().sum()


# In[11]:


print(df.fruit_label.unique())
print(df.fruit_name.unique())


# To make the results easier to understand, we first establish a mapping from fruit label value to fruit name.

# In[12]:


# create a mapping from a mapping from fruit label value to fruit name
lookup_fruit_name = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))
lookup_fruit_name


# #### Exploratory data analysis
# 
# It is now time to experiment with the data and make some visualizations.
# 

# In[13]:


sns.pairplot(df[['fruit_name','mass','width','height']],hue='fruit_name')


# #### What we observe from the figures?
# 
# 1. Mandarin has both lower mass and height. It also has the lower average widths.
# 2. Orange has higher average masses and widths.
# 3. There is a **clear separation** of lemon from the other both width-height plot and height-mass plot.
# 4. What else can we observe from the figures?
# 

# In[14]:


df.groupby('fruit_name').describe()[['mass']]


# In[15]:


df.groupby('fruit_name').describe()[['height']]


# In[16]:


df.groupby('fruit_name').describe()[['width']]


# #### Preprocessing: Train Test Split.
# 
# Because training and testing on the same data is inefficient, we partition the data into two sets: **training and testing**. 
# 
# To split the data, we use the `train_test_split` function. 
# 
# The split percentage is determined by the optional parameter `test_size.` The default values are 75/25% train and test data.
# 
# The `random state` parameter ensures that the data is split in the same way each time the program is executed. 
# 
# Because we are training and testing on distinct sets of data, the testing accuracy will be a better indication of how well the model will perform on new data.
# 
# 

# In[17]:


# Train Test Split

X = df[['height', 'width', 'mass']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[18]:


#shape of train and test data
print(X_train.shape)
print(X_test.shape)


# In[19]:


#shape of new y objects
print(y_train.shape)
print(y_test.shape)


# #### Training and Predictions
# 
# Scikit-learn is divided into modules so that we may quickly import the classes we need. 
# 
# Import the `KNeighborsClassifer` class from the `neighbors` module. 
# 
# Instantiate the estimator (a model in scikit-learn is referred to as a **estimator**). Because their major function is to estimate unknown quantities, we refer to the model as an estimator.
# 
# In our example, we have generated an instance (`knn`) of the class `KNeighborsClassifer`, which means we have constructed an object called 'knn' that knows how to perform KNN classification once the data is provided. 
# 
# The **tuning parameter/hyper parameter** (k) is the parameter `n_ neighbors`. All other parameters are set to default.
# 
# The `fit` method is used to train the model using training data (X train,y train), while the `predict` method is used to test the model using testing data (X test). 

# In this example, we take `n_neighbors` or k = 5.

# In[20]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)


# #### Train the classifier (fit the estimator) using the training data¶
# 
# 
# We then train the classifier by passing the training set data in **X_train** and the labels in **y_train** to the classifier's fit method.

# In[21]:


knn.fit(X_train,y_train)


# #### Estimate the accuracy of the classifier on future data, using the test data 
# 
# 
# Remember that the KNN classifier has not seen any of the fruits in the **test set** during the training phase.
# 
# To do this, we use the score method for the classifier object.
# This takes the points in the test set as input and calculates the **accuracy**. 
# 
# The **accuracy** is defined as the proportion of points in the test set whose true label was correctly predicted by the classifier.

# In[22]:


knn.score(X_test, y_test)

print('Accuracy:', knn.score(X_test, y_test))


# We obtain a classficiation rate of 53.3%, considered as good accuracy. 
# 
# Can we further improve the accuracy of the KNN algorithm? 

# #### Use the trained KNN classifier model to classify new, previously unseen objects
# 
# So, here for example. We are entering the mass, width, and height for a hypothetical piece of fruit that is pretty small. 
# 
# And if we ask the classifier to predict the label using the predict method.
# 
# We can see that the output says that it is a mandarin.
# 
# for example: a small fruit with a mass of 20 g, a width of 4.5 cm and a height of 5.2 cm

# In[23]:


X_train.columns


# In[24]:


#sample1 = pd.DataFrame({'height':[5.2], 'width':[4.5],'mass':[20]})
# Notice we use the same column as the X training data (or X test sate)
sample1 = pd.DataFrame([[5.2,4.5,20]],columns = X_train.columns)
fruit_prediction = knn.predict(sample1)

print('The prediction is:', lookup_fruit_name[fruit_prediction[0]])


# Here another example

# In[25]:


#sample2 = pd.DataFrame({'height':[6.8], 'width':[8.5],'mass':[180]})

sample2 = pd.DataFrame([[6.8,8.5,180]],columns = X_train.columns)
fruit_prediction = knn.predict(sample2)

print('The prediction is:', lookup_fruit_name[fruit_prediction[0]])


# In[ ]:





# **Exercise** 
# 
# 1. Create a DataFrame comparing the y_test and the predictions from the model.
# 
# 2. Confirm that the accuracy is the same as obtained by knn.score(X_test, y_test).

# Here is the list of predictions of the test set obtained from the model.

# In[26]:


#fruit_prediction = knn.predict([[20,4.5,5.2]])
y_pred = knn.predict(X_test)


# #### Calculating the agorithm accuracy

# In[27]:


# the accuracy of the test set

# https://stackoverflow.com/questions/59072143/pandas-mean-of-boolean
((knn.predict(X_test) == y_test.to_numpy())*1).mean()
(knn.predict(X_test) == y_test.to_numpy()).mean()


# In[28]:


# the accuracy of the original data set

# (knn.predict(X) == y.to_numpy()).mean()


# In[29]:


# the accuracy of the training set

(knn.predict(X_train) == y_train.to_numpy()).mean()


# #### Comparison of the observed and predicted values from both training and test data sets

# In[30]:


# Printing out the observed and predicted values of the test set

for i in range(len(X_test)):
    print('Observed: ', lookup_fruit_name[y_test.iloc[i]], 'vs Predicted:',    lookup_fruit_name[knn.predict(X_test)[i]])


# In[31]:


print(X_test.index)
y_test.index


# In[32]:


pd.DataFrame({'observed': y_test ,'predicted':knn.predict(X_test)}).set_index(X_test.index) 


# In[33]:


pd.DataFrame({'observed': y_train ,'predicted':knn.predict(X_train)}).set_index(X_train.index).head()


# In[34]:


pd.DataFrame({'observed': y ,'predicted':knn.predict(X)}).set_index(X.index).head() 


# In[35]:


df_result = df
df_result['predicted_label'] = knn.predict(X)
df_result.head()


# #### Visualize the decision regions of a classifier
# 
# 
# After a classifier has been trained on training data, a classification model is created. What criteria does your machine learning classifier consider when deciding which class a sample belongs to? Plotting a decision region can provide some insight into the decision made by your ML classifier.
# 
# 
# A **decision region** is a region in which a classifier predicts the same class label for data. 
# 
# The boundary between areas of various classes is known as the **decision boundary**. 
# 
# The plot decision regions function in `mlxtend` is a simple way to plot decision areas. We can also use mlxtend to plot  decision regions of **Logistic Regression, Random Forest, RBF kernel SVM, and Ensemble classifier**.
# 
# **Important Note** for the 2D scatterplot, we can **only visualize 2 features** at a time. So, if you have a 3-dimensional (or more than 3) dataset, it will essentially be a **2D slice through this feature space with fixed values for the remaining feature(s)**.

# In[36]:


from mlxtend.plotting import plot_decision_regions


# In[84]:


# Importing a dataset
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
df = pd.read_table(url)  

# Train Test Split

X = df[['height', 'width', 'mass']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Instantiate the estimator
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Training the classifier by passing in the training set X_train and the labels in y_train
knn.fit(X_train,y_train)

# Predicting labels for unknown data
y_pred = knn.predict(X_test)


# In[85]:


X_train.describe()


# #### Decision regions for two features, height and mass.

# In[38]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train.values, y_train.values)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 2 = width = value
value=7

# Plot training sample with feature 2 = width = value +/- width
width=0.5

fig = plot_decision_regions(X_train.values, y_train.values, clf=knn,
              feature_index=[0,2],                        #these one will be plotted  
              filler_feature_values={1: value},  #these will be ignored
              filler_feature_ranges={1: width})
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (mass)')
ax.set_title('Feature 3 (width) = {}'.format(value))


# In[39]:


# Points to be included in the plot above, 
# i.e. those sample points with width between (value - width, value + width)

# X_train

X_train.head()
#X_train.head()
print(X_train[(X_train.width < value + width) & (X_train.width > value - width)].sort_values(by = 'height'))
print(y_train[(X_train.width < value + width) & (X_train.width > value - width)])


# In[40]:


#df.head()
#X_train.head()
#df[(df.width < 7.2) & (df.width > 6.8)].sort_values(by = 'height')


# #### Decision regions for two features, height and width.

# In[41]:


#X_train.describe()


# In[42]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train.values, y_train.values)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature 3 = mass = value +/- width
width=30

fig = plot_decision_regions(X_train.values, y_train.values, clf=knn,
              feature_index=[0,1],               #these one will be plotted  
              filler_feature_values={2: value},  #these will be ignored
              filler_feature_ranges={2: width})
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (width)')
ax.set_title('Feature 3 (mass) = {}'.format(value))


# In[43]:


#print(X_train.head())
#print(y_train)


# In[44]:


##### Convert pandas DataFrame to Numpy before applying classification
##### 
#### weights{‘uniform’, ‘distance’} 

#### ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.

#### ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.


X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()


clf = KNeighborsClassifier(n_neighbors = 5,weights='distance')
clf.fit(X_train_np, y_train_np)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature = mass = value +/- width
width=100

fig = plot_decision_regions(X_train_np, y_train_np, clf=clf,
              feature_index=[0,1],               #these one will be plotted  
              filler_feature_values={2: value},  #these will be ignored
              filler_feature_ranges={2: width})
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (width)')
ax.set_title('Feature 3 (mass) = {}'.format(value))


# In[45]:


clf.predict(X_train_np)
y_train_np
X_train.describe()


# In[46]:


# X_train_np = X_train.iloc[:,0:2].to_numpy()
# y_train_np = y_train.to_numpy()

# X_train_np.shape


# #### Applying KNN classfication with only two features

# In[47]:


##### Convert pandas DataFrame to Numpy before applying classification

X_train_np = X_train.iloc[:,0:2].to_numpy()
y_train_np = y_train.to_numpy()


clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train_np, y_train_np)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature = mass = value +/- width
width=20

fig = plot_decision_regions(X_train_np, y_train_np, clf=clf)
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (width)')
ax.set_title('Classification problem')


# In[48]:


##### Convert pandas DataFrame to Numpy before applying classification

X_train_np = X_train.iloc[:,[0,2]].to_numpy()
y_train_np = y_train.to_numpy()


clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train_np, y_train_np)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature = mass = value +/- width
width=20

fig = plot_decision_regions(X_train_np, y_train_np, clf=clf)
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (mass)')
ax.set_title('Classification problem')


# #### Width-Mass visualization

# In[49]:


#print(X_train.head())
#print(X_train.values)
#X_train.to_numpy()

#print(type(X_train.values))
#print(type(X_train.to_numpy()))

#print(X_train.values.shape)
#print(X_train.to_numpy().shape)


# In[50]:


X_train.describe()


# In[51]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train.values, y_train.values)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 1 = height = value
value=7.6
# Plot training sample with feature 1 = height = value +/- width
width=2.6

fig = plot_decision_regions(X_train.values, y_train.values, clf=knn,
              feature_index=[1,2],               #these one will be plotted  
              filler_feature_values={0: value},  #these will be ignored
              filler_feature_ranges={0: width})
ax.set_xlabel('Feature 1 (width)')
ax.set_ylabel('Feature 2 (mass)')
ax.set_title('Feature 3 (height) = {}'.format(value))


# In[ ]:





# #### Width-height visualization

# In[52]:


#### Applying KNN classfication with only two features

X_train[['width','height']].to_numpy()
y_train.to_numpy()

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train[['width','height']].to_numpy(), y_train.to_numpy())

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature 3 = mass = value +/- width
width=50

fig = plot_decision_regions(X_train[['width','height']].to_numpy(), y_train.to_numpy(), clf=knn)
ax.set_xlabel('Feature 1 (width)')
ax.set_ylabel('Feature 2 (mass)')
ax.set_title('Feature 3 (height) = {}'.format(value))


# In[53]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train.values, y_train.values)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature 3 = mass = value +/- width
width=100

fig = plot_decision_regions(X_train.values, y_train.values, clf=knn,
              feature_index=[1,0],               #these one will be plotted  
              filler_feature_values={2: value},  #these will be ignored
              filler_feature_ranges={2: width})
ax.set_xlabel('Feature 1 (width)')
ax.set_ylabel('Feature 2 (height)')
ax.set_title('Feature 3 (mass) = {}'.format(value))


# #### Visualize (from scratch) the decision regions of a classifier

# In[54]:


from matplotlib.colors import ListedColormap
from sklearn import neighbors


# In[55]:


#X = df[['height', 'width']].to_numpy()
#y = df['fruit_label'].to_numpy()


# In[56]:


# The code below has been modified based on https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
#X = iris.data[:, :2]
#y = iris.target

X = df[['height', 'width']].to_numpy()
y = df['fruit_label'].to_numpy()

n_neighbors = 5

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue","green"])
cmap_bold = ["darkorange", "c", "darkblue","darkgreen"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue= df.fruit_name.to_numpy(),
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel('height')
    plt.ylabel('width')

plt.show()


# In[57]:


#https://stackoverflow.com/questions/52952310/plot-decision-regions-with-error-filler-values-must-be-provided-when-x-has-more
# You can use PCA to reduce your data multi-dimensional data to two dimensional data. Then pass the obtained result in plot_decision_region and there will be no need of filler values.

from mlxtend.plotting import plot_decision_regions


# In[58]:


# Decision region of the training set

X = df[['height', 'width']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train,y_train)

plot_decision_regions(X_train.to_numpy(), y_train.to_numpy(), clf=clf, legend=2)


# In[59]:


# Decision region of the test set

plot_decision_regions(X_test.to_numpy(), y_test.to_numpy(), clf=clf, legend=2)


# In[ ]:





# ## Model Evaluation Metrics in Machine Learning
# 
# Machine learning has become extremely popular in recent years. Machine learning is used to infer new situations from past data, and there are far too many machine learning algorithms to choose from.
# 
# Machine learning techniques such as 
# 
# * linear regression, 
# 
# * logistic regression, 
# 
# * decision tree, 
# 
# * Naive Bayes, K-Means, and 
# 
# * Random Forest 
# 
# are widely used. 
# 
# When it comes to predicting data, we **do not use just one algorithm**. Sometimes we use multiple algorithms and then proceed with the one that gives the best data predictions. 
# 
# ### How can we figure out which algorithm is the most effective? 
# 
# Model evaluation metrics allow us to evaluate the accuracy of our trained model and track its performance. 
# 
# Model evaluation metrics, which distinguish adaptive from non-adaptive machine learning models, indicate how effectively the model generalizes to new data.
# 
# We could improve the overall predictive power of our model before using it for production on unknown data by using different performance evaluation metrics. 
# 
# Choosing the right metric is very important when evaluating machine learning models. Machine learning models are evaluated using a variety of metrics in different applications. Let us look at the metrics for evaluating the performance of a machine learning model. 
# 
# This is a critical phase in any data science project as it aims to estimate the generalization accuracy of a model for future data.

# **Evaluation Metrics For Regression Models**: image from enjoyalgorithms
# ![Evaluation Metrics For Regression Models from enjoyalgorithms](https://www.enjoyalgorithms.com/static/evaluation-metrics-regression-models-cover-6422d3e49173675d75d121740c04d450.jpg)

# **Evaluation Metrics For Classification Models**: image from enjoyalgorithms
# ![Evaluation Metrics For Classification Models from enjoyalgorithms](https://www.enjoyalgorithms.com/static/evaluation-metrics-classification-models-cover-4f403c2e47e719b4389b4b2d05d71c34.jpg)

# #### Regression Related Metrics
# 
# The most common measures for evaluating a regression model (as used in our previous chapter) are:
# 
# * **Mean Absolute Error (MAE)**: The average of the difference between the actual and anticipated values is the Mean Absolute Error. It determines how close the predictions are to the actual results. The better the model, the lower the MAE.
# 
# * **Mean Squared Error (MSE)**: The average of the square of the difference between the actual and predicted values is calculated by MSE.
# 
# * **R2 score**: The proportion of variance in Y that can be explained by X is called the R2 score.

# #### Classification Metrics
# 
# 1. **Confusion Matrix (Accuracy, Sensitivity, and Specificity)**
# 
# A confusion matrix contains the results of any binary testing that is commonly used to describe the classification model's performance.
# 
# In a binary classification task, there are only two classes to categorize, preferably a **positive class** and a **negative class**. 
# 
# Let us take a look at the metrics of the confusion matrix.
# 
# * **Accuracy**: indicates the overall accuracy of the model, i.e., the percentage of all samples that were correctly identified by the classifier. Use the following formula to calculate accuracy: 
# (TP +TN)/(TP +TN+ FP +FN).
# 
#     * True Positive (TP): This is the number of times the classifier successfully predicted the positive class to be positive.
# 
#     * True Negative (TN): The number of times the classifier correctly predicts the negative class as negative.
# 
#     * False Positive (FP): This term refers to the number of times a classifier incorrectly predicts a negative class as positive.
# 
#     * False Negative (FN): This is the number of times the classifier predicts the positive class as negative.
# 
# 
# * **The misclassification rate**: tells you what percentage of predictions were incorrect. It is also called classification error. You can calculate it with (FP +FN)/(TP +TN+ FP +FN) or (1-accuracy).
# 
# * **Sensitivity (or Recall)**: It indicates the proportion of all positive samples that were correctly predicted to be positive by the classifier. It is also referred to as **true positive rate (TPR), sensitivity, or probability of detection**. To calculate recall, use the following formula: TP /(TP +FN).
# 
# * **Specificity**: it indicates the proportion of all negative samples that are correctly predicted to be negative by the classifier. It is also referred to as the **True Negative Rate (TNR)**. To calculate the specificity, use the following formula: TN /(TN +FP).
# 
# 2. **Precision**: When there is an **imbalance between classes**, accuracy can become an unreliable metric for measuring our performance. Therefore, we also need to address class-specific performance metrics. Precision is one such metric, defined as **positive predictive values** (Proportion of predictions as positive class were actually positive). To calculate precision, use the following formula: TP/(TP+FP).
# 
# 3. **F1-score**: it combines precision and recall in a single measure. Mathematically, it is the harmonic mean of Precision and Recall. It can be calculated as follows:
# 
# $${\displaystyle F_{1}={\frac {2}{\mathrm {recall^{-1}} +\mathrm {precision^{-1}} }}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}={\frac {\mathrm {tp} }{\mathrm {tp} +{\frac {1}{2}}(\mathrm {fp} +\mathrm {fn} )}}}.$$
# 
# In a perfect world, we would want a model that has a precision of 1 and a recall of 1. This means an F1 score of 1, i.e. 100% accuracy, which is often not the case for a machine learning model. So we should try to achieve a higher precision with a higher recall value. 

# **Confusion Matrix for Binary Classification**: image from https://towardsdatascience.com/
# ![Confusion Matrix for Binary Classification image from https://towardsdatascience.com/](https://miro.medium.com/max/700/1*fxiTNIgOyvAombPJx5KGeA.png)
# 
# 

# **Confusion Matrix for Multi-class Classification**: image from https://towardsdatascience.com/
# ![Confusion Matrix for Multi-class Classification image from https://towardsdatascience.com/](https://miro.medium.com/max/700/1*yH2SM0DIUQlEiveK42NnBg.png)

# In[60]:


# Importing a dataset
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
df = pd.read_table(url)  

# Train Test Split

X = df[['height', 'width', 'mass']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Instantiate the estimator
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Training the classifier by passing in the training set X_train and the labels in y_train
knn.fit(X_train,y_train)

# Predicting labels for unknown data
y_pred = knn.predict(X_test)


# In[61]:


#importing confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)


# In[62]:


# Confusion Matrix visualization.

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
disp.plot()


# In[63]:


pd.DataFrame({'observed': y_test ,'predicted':knn.predict(X_test)}).set_index(X_test.index).sort_values(by = 'observed') 


# Unlike the binary classification, there are no positive or negative classes here. 
# 
# At first glance, it might be a little difficult to find TP, TN, FP, and FN since there are no positive or negative classes, but it's actually pretty simple. 
# 
# What we need to do here is find TP, TN, FP and FN for each and every class. For example, let us take the **Apple class**. Let us look at what values the metrics have in the confusion matrix.
# (DO NOT FORGET TO TRANSPOSE)
# 
# * TP = 3
# 
# * TN = (1 + 3 + 2 + 1 + 1) = 8 (the sum of the numbers in rows 2-4 and columns 2-4)
# 
# * FP = (0 + 3 + 0) = 3
# 
# * FN = (0 + 0 + 1) = 1
# 
# Now that we have all the necessary metrics for the Apple class from the confusion matrix, we can calculate the performance metrics for the Apple class. For example, the class Apple has
# 
# * Precision = 3/(3+3) = 0.5
# 
# * Recall = 3/(3+1) = 0.75
# 
# * F1-score = 0.60
# 
# 
# In a similar way, we can calculate the measures for the other classes. Here is a table showing the values of each measure for each class.

# In[64]:


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))


# Now we can do more with these measures. We can combine the F1 score of each class to get a single measure for the entire model. There are several ways to do this, which we will now look at.
# 
# * **Macro F1**
# This is the macro-averaged F1 score. It calculates the metrics for each class separately and then takes the unweighted average of the measures. As we saw in the figure "Precision, recall and F1 score for each class",
# 
# * **Weighted F1**
# The final value is the weighted mean F1 score. Unlike Macro F1, this uses a weighted mean of the measures. The weights for each class are the total number of samples in that class. Since we had 4 apples, 1 mandarin, 8 oranges, and 3 lemons,

# We obtain a classficiation rate of 53.3%, considered as good accuracy. 
# 
# Can we further improve the accuracy of the KNN algorithm? 

# In our example, we have created an instance ('knn') of the class 'KNeighborsClassifer,' which means we have constructed an object called 'knn' that knows how to perform KNN classification once the data is provided. 
# 
# The tuning parameter/hyper parameter (K) is the parameter **n_neighbors**. All other parameters are set to default values.
# 
# **Exercises**
# 
# 1. Fit the model and test it for different values for K (from 1 to 5) using a for loop and record the KNN's testing accuracy of the KNN in a variable.
# 
# 2. Plot the relationship between the values of K and the corresponding testing accuracy.
# 
# 3. Select the optimal value of K that gives the highest testing accuray.
# 
# 4. Compare the results between the optimal value of K and K = 5.

# In[65]:


# Importing a dataset
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
df = pd.read_table(url)  

# Train Test Split

X = df[['height', 'width', 'mass']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Instantiate the estimator
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

k_range = range(1,11)

train_accuracy = {}
train_accuracy_list = []

test_accuracy = {}
test_accuracy_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # Training the classifier by passing in the training set X_train and the labels in y_train
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[k] = knn.score(X_train, y_train)
    train_accuracy_list.append(knn.score(X_train, y_train))
    
    
    test_accuracy[k] = knn.score(X_test,y_test)
    test_accuracy_list.append(knn.score(X_test,y_test))


# In[66]:


df_output = pd.DataFrame({'k':k_range,
                          'train_accuracy':train_accuracy_list,
                          'test_accuracy':test_accuracy_list
                         })

(
    ggplot(df_output) 
    + geom_line(aes(x = 'k',  y = 'train_accuracy',color='"training accuracy"'))
    + geom_line(aes(x = 'k',  y = 'test_accuracy',color='"testing accuracy"'))
    + labs(x='k (n_neighbors)', y='overall accuracy')
    + scale_color_manual(values = ["blue", "green"], # Colors
        name = " ")
)


# In[67]:


y_test.shape
np.sqrt(15)


# In[68]:


import seaborn as sns
import matplotlib.pyplot as plt

hm = sns.heatmap(df.corr(), annot = True)
hm.set(title = "Correlation matrix of insurance data\n")
plt.show()


# In[69]:


sns.pairplot(df[['fruit_name','color_score','height']],hue='fruit_name')


# In[70]:


# Importing a dataset
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
df = pd.read_table(url)  

# Train Test Split

X = df[['height', 'color_score']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Instantiate the estimator
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

k_range = range(1,11)

train_accuracy = {}
train_accuracy_list = []

test_accuracy = {}
test_accuracy_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # Training the classifier by passing in the training set X_train and the labels in y_train
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[k] = knn.score(X_train, y_train)
    train_accuracy_list.append(knn.score(X_train, y_train))
    
    
    test_accuracy[k] = knn.score(X_test,y_test)
    test_accuracy_list.append(knn.score(X_test,y_test))


# In[71]:


df_output = pd.DataFrame({'k':k_range,
                          'train_accuracy':train_accuracy_list,
                          'test_accuracy':test_accuracy_list
                         })

(
    ggplot(df_output) 
    + geom_line(aes(x = 'k',  y = 'train_accuracy',color='"training accuracy"'))
    + geom_line(aes(x = 'k',  y = 'test_accuracy',color='"testing accuracy"'))
    + labs(x='k (n_neighbors)', y='overall accuracy')
    + scale_color_manual(values = ["blue", "green"], # Colors
        name = " ")
)


# In[72]:


##### Convert pandas DataFrame to Numpy before applying classification

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

clf = KNeighborsClassifier(n_neighbors = 4)
clf.fit(X_train_np, y_train_np)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature = mass = value +/- width
width=20

fig = plot_decision_regions(X_test_np, y_test_np, clf=clf)
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (color_scale)')
ax.set_title('Classification problem')


# In[73]:


clf.predict([[ 8.5 ,  0],
       [9.2 ,  0]])


# In[ ]:





# ## Scaling
# 
# Because KNN uses the Euclidean distance between points to determine the closest neighboring points, all of the data must be on the same scale.

# In[74]:


df.head()


# In[75]:


#df[['mass','width','height','color_score']]


# In[76]:


from sklearn.preprocessing import MinMaxScaler

X = df[['mass','width','height','color_score']]
y = df['fruit_label']

scaler = MinMaxScaler()

scaled_X = pd.DataFrame(scaler.fit_transform(X),
                        columns=X.columns)


# In[77]:


scaled_X.head()


# In[78]:


scaled_df = scaled_X
scaled_df['fruit_label'] = df[['fruit_label']]
#scaled_df

hm = sns.heatmap(scaled_df.corr(), annot = True)
hm.set(title = "Correlation matrix of insurance data\n")
plt.show()


# In[79]:


sns.pairplot(scaled_df[['fruit_label','color_score','height']],hue='fruit_label')


# In[ ]:





# In[80]:


# Importing a dataset
#url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
#df = pd.read_table(url)  

# Train Test Split

X = scaled_df[['height', 'color_score']]
y = scaled_df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Instantiate the estimator
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

k_range = range(1,11)

train_accuracy = {}
train_accuracy_list = []

test_accuracy = {}
test_accuracy_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # Training the classifier by passing in the training set X_train and the labels in y_train
    knn.fit(X_train,y_train)
    
    # Compute accuracy on the training set
    train_accuracy[k] = knn.score(X_train, y_train)
    train_accuracy_list.append(knn.score(X_train, y_train))
    
    
    test_accuracy[k] = knn.score(X_test,y_test)
    test_accuracy_list.append(knn.score(X_test,y_test))


# In[81]:


train_accuracy
print(test_accuracy)


# In[82]:


scaled_df_output


# In[ ]:


scaled_df_output = pd.DataFrame({'k':k_range,
                          'train_accuracy':train_accuracy_list,
                          'test_accuracy':test_accuracy_list
                         })

(
    ggplot(scaled_df_output) 
    + geom_line(aes(x = 'k',  y = 'train_accuracy',color='"training accuracy"'))
    + geom_line(aes(x = 'k',  y = 'test_accuracy',color='"testing accuracy"'))
    + labs(x='k (n_neighbors)', y='overall accuracy')
    + scale_color_manual(values = ["blue", "green"], # Colors
        name = " ")
)


# In[ ]:


##### Convert pandas DataFrame to Numpy before applying classification

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

clf = KNeighborsClassifier(n_neighbors = 4)
clf.fit(X_train_np, y_train_np)

# Plotting decision regions
fig, ax = plt.subplots()

# Decision region for feature 3 = mass = value
value=160
# Plot training sample with feature = mass = value +/- width
width=20

fig = plot_decision_regions(X_test_np, y_test_np, clf=clf)
ax.set_xlabel('Feature 1 (height)')
ax.set_ylabel('Feature 2 (color_scale)')
ax.set_title('Classification problem')


# In[ ]:


test_accuracy


# In[ ]:





# In[ ]:




