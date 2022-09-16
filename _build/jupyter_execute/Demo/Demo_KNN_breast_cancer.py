#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Plotting the classification results
from mlxtend.plotting import plot_decision_regions


# In[2]:



# Importing the dataset
dataset = load_breast_cancer() 

# Converting to pandas DataFrame
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)

print("Total samples in our dataset is: {}".format(df.shape[0]))

# Describe the dataset
df.describe()


# In[4]:



# Selecting the features
features = ['mean perimeter', 'mean texture']
x = df[features]

# Target variable
y = df['target']


# In[7]:



# Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 25 )

# Fitting KNN Classifier to the Training set
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(x_train, y_train)

# Predicting the results
y_pred = model.predict(x_test)

# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('KNN Classification Accuracy of the model: {:.2f}%'.format(accuracy*100))


# In[8]:



# Plotting the decision boundary
plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
plt.title("Decision boundary using KNN Classification (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")


# In[ ]:




