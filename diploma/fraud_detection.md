

```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
import sys
```


```python
df = pd.read_csv('creditcard.csv')
```


```python
print(df.head())
```

         Time         V1         V2         V3         V4         V5        V6  \
    0   28755 -30.552380  16.713389 -31.103685   6.534984 -22.105532 -4.977692   
    1   28726 -29.876366  16.434525 -30.558697   6.505862 -21.665654 -4.940356   
    2   28692 -29.200329  16.155701 -30.013712   6.476731 -21.225810 -4.902997   
    3  102572 -28.709229  22.057729 -27.855811  11.845013 -18.983813  6.474115   
    4   28658 -28.524268  15.876923 -29.468732   6.447591 -20.786000 -4.865613   
    
              V7         V8         V9  ...       V22       V23       V24  \
    0 -20.371514  20.007208  -3.565738  ... -2.288686 -1.460544  0.183179   
    1 -20.081391  19.587773  -3.591491  ... -2.232252 -1.412803  0.178731   
    2 -19.791248  19.168327  -3.617242  ... -2.175815 -1.365104  0.174286   
    3 -43.557242 -41.044261 -13.320155  ...  8.316275  5.466230  0.023854   
    4 -19.501084  18.748872  -3.642990  ... -2.119376 -1.317450  0.169846   
    
            V25       V26       V27       V28  Amount  Class  Fraud  
    0  2.208209 -0.208824  1.232636  0.356660   99.99      1   True  
    1  2.156042 -0.209385  1.255649  0.364530   99.99      1   True  
    2  2.103868 -0.209944  1.278681  0.372393   99.99      1   True  
    3 -1.527145 -0.145225 -5.682338 -0.439134    0.01      1   True  
    4  2.051687 -0.210502  1.301734  0.380246   99.99      1   True  
    
    [5 rows x 32 columns]
    


```python
X_all = df.drop(['Class', 'Fraud'],1)
y_all = df['Fraud']
```


```python
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
```

    Processed feature columns (30 total features):
    ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    


```python

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.2,
                                                    random_state = 4,
                                                    stratify = y_all)
X_train.shape
```




    (800, 30)




```python
from sklearn import svm


svc_classifier = svm.SVC()
svc_classifier.fit(X_train, y_train)

y_predict = svc_classifier.predict(X_test)
```


```python
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_predict))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
```

    [[91 11]
     [24 74]]
    Accuracy: 0.825
    


```python
metrics.confusion_matrix(y_test, y_predict, labels=[True, False])
```




    array([[74, 24],
           [11, 91]], dtype=int64)




```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
```

                  precision    recall  f1-score   support
    
           False       0.79      0.89      0.84       102
            True       0.87      0.76      0.81        98
    
        accuracy                           0.82       200
       macro avg       0.83      0.82      0.82       200
    weighted avg       0.83      0.82      0.82       200
    
    


```python
print(metrics.balanced_accuracy_score(y_test, y_predict))
```

    0.8236294517807123
    


```python
print(metrics.log_loss(y_test, y_predict))
```

    6.04432984696803
    


```python

```
