#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
import sys


# In[3]:


df = pd.read_csv('creditcard.csv')


# In[4]:


print(df.head())


# In[6]:


X_all = df.drop(['Class', 'Fraud'],1)
y_all = df['Fraud']


# In[7]:


#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[8]:



#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.2,
                                                    random_state = 4,
                                                    stratify = y_all)
X_train.shape


# In[9]:


from sklearn import svm


svc_classifier = svm.SVC()
svc_classifier.fit(X_train, y_train)

y_predict = svc_classifier.predict(X_test)


# In[10]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_predict))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))


# In[11]:


metrics.confusion_matrix(y_test, y_predict, labels=[True, False])


# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[13]:


print(metrics.balanced_accuracy_score(y_test, y_predict))


# In[14]:


print(metrics.log_loss(y_test, y_predict))


# In[ ]:




