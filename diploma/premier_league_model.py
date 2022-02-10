#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[45]:


game_data = pd.read_csv('dataset_final.csv')
#game_data = game_data[['FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR','HomeTeam', 'AwayTeam','Referee']]
game_data.head()


# In[46]:


game_data.shape


# In[47]:


game_data.size


# In[48]:


game_data.count()


# In[49]:


game_data['FTR'].value_counts()


# In[50]:


# Visualising distribution of data
from pandas.plotting import scatter_matrix
scatter_matrix(game_data[['HS','AS','HST','AST','HF', 'AF', 'HC','AC']], figsize=(10,10))


# In[51]:


# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = game_data.drop(['FTR','Div','Referee','HomeTeam', 'AwayTeam'],1)
# X_all = game_data.drop(['FTR','HF','AF','HC','AC','HY','AY','HR','AR','Referee','Div'],1)
y_all = game_data['FTR']


# In[53]:


X_all.head()


# In[54]:


# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
# cols = [['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST']]
for col in cols:
    X_all[col] = scale(X_all[col])


# In[55]:


X_all.HTR = X_all.HTR.astype('str')
# X_all.HomeTeam = X_all.HomeTeam.astype('str')
# X_all.AwayTeam = X_all.AwayTeam.astype('str')
X_all.head()


# In[56]:


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


# In[57]:


# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())


# In[58]:



#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.2,
                                                    random_state = 4,
                                                    stratify = y_all)
X_train.shape


# In[59]:


from sklearn import svm


svc_classifier = svm.SVC(kernel='poly', gamma='auto', C=10)
svc_classifier.fit(X_train, y_train)

y_predict = svc_classifier.predict(X_test)


# In[60]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))


# In[71]:


from sklearn import metrics
metrics.confusion_matrix(y_test, y_predict, labels=['A','H','D'])


# In[62]:


metrics.accuracy_score(y_test, y_predict)


# In[65]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


# In[66]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[67]:


print(classification_report(y_test, y_pred))


# In[70]:


from sklearn import metrics
metrics.confusion_matrix(y_test, y_pred, labels=['A','H','D'])


# In[ ]:




