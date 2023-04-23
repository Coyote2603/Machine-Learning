#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# #### Loading the Iris dataset: We load the iris dataset using the load_iris() function from sklearn.datasets. This dataset is a popular multi-class classification dataset with 150 samples and four features: sepal length, sepal width, petal length, and petal width. The target variable represents the class labels (species) of the iris flowers. 

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


X = iris.data
y = iris.target


# ## Linear Discriminant Analysis

# #### Perform LDA with p=1 

# In[4]:


lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)


# #### Visualize the projected data 

# In[5]:


plt.figure(figsize=(8, 6))
plt.scatter(X_lda[y == 0], np.zeros_like(X_lda[y == 0]), color='r', label=iris.target_names[0])
plt.scatter(X_lda[y == 1], np.zeros_like(X_lda[y == 1]), color='g', label=iris.target_names[1])
plt.scatter(X_lda[y == 2], np.zeros_like(X_lda[y == 2]), color='b', label=iris.target_names[2])
plt.xlabel('LD1')
plt.legend()
plt.show()


# #### In this code, we first load the Iris dataset using load_iris() from sklearn.datasets. We then create an instance of LinearDiscriminantAnalysis from sklearn.discriminant_analysis, setting n_components to 1 to specify that we want to project the data onto a 1-dimensional subspace. Next, we use the fit_transform() method of the LDA object to perform LDA on the input data X with the corresponding target labels y, and obtain the projected data X_lda. Finally, we visualize the projected data using a scatter plot, where each species is represented by a different color. The x-axis represents the first linear discriminant (LD1), which is the 1-dimensional subspace onto which the data is projected. 

# #### In practice, it's important to evaluate the performance of the LDA model using appropriate evaluation techniques, such as cross-validation, and to validate the assumptions of normality and equal covariance matrices before applying LDA to real-world datasets. 

# #### Perform LDA with p>1 

# In[6]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)


# #### Plot the transformed data 

# In[7]:


plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'D']
for c, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(X_lda[y == c, 0], X_lda[y == c, 1], c=color, marker=marker, label=iris.target_names[c])
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.title('Linear Discriminant Analysis on Iris Dataset')
plt.show()


# #### In this example, we first load the Iris dataset using the load_iris function from scikit-learn. We then perform Linear Discriminant Analysis using the LinearDiscriminantAnalysis class from scikit-learn, specifying the number of components to be 2 to obtain a reduced-dimensional representation of the data. Finally, we plot the transformed data using matplotlib to visualize the results, with different colors and markers representing the different classes of Iris flowers. 

# ## Quadratic Discriminant Analysis 

# #### Split the data into training and testing sets: Next, we split the data into training and testing sets using train_test_split() function. We use 80% of the data for training and 20% for testing, and set the random_state parameter to 42 for reproducibility. 

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Initialize QDA model 

# In[9]:


qda = QuadraticDiscriminantAnalysis()


# #### Fit the QDA model to the training data: We fit the QDA model to the training data using the fit() method. This step involves estimating the parameters of the QDA model from the training data. 

# In[10]:


qda.fit(X_train, y_train)


# #### Once the QDA model is trained, we use it to make predictions on the testing data using the predict() method.

# In[11]:


y_pred = qda.predict(X_test)


# In[12]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# #### The QDA model on the testing data is 96.67% accurate. This gives us an estimate of how well our model is performing in terms of classification accuracy. 

# # Generate confusion matrix

# In[13]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[14]:


sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# #### Class-wise Scatter Plots: Scatter plots can be used to visualize the distribution of data points for each class separately in the feature space. This can help identify any overlapping regions between classes and assess the discriminative power of the features. 

# In[15]:


iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = iris.target


# In[16]:


sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', style='species', data=iris_df)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Class-wise Scatter Plot')
plt.legend()
plt.show()


# # Naïve Bayes

# In[17]:


gnb = GaussianNB()


# In[18]:


gnb.fit(X_train, y_train)


# In[19]:


y_pred = gnb.predict(X_test)


# In[20]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[22]:


confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)


# #### The accuracy represents the percentage of correctly predicted samples in the test set. The classification report provides detailed metrics such as precision, recall, and F1-score for each class. The confusion matrix displays the number of correct and incorrect predictions for each class, which can help assess the performance of the model in terms of false positives, false negatives, true positives, and true negatives 
