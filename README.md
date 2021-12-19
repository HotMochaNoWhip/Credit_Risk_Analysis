# Credit_Risk_Analysis

---

## Overview
- Create training and test groups from a given data set.
- Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
- Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
- Compare the advantages and disadvantages of each supervised learning algorithm.
- Determine which supervised learning algorithm is best used for a given data set or scenario.
- Use ensemble and resampling techniques to improve model performance.


---

## Analysis

In this project we used data from Credit Loans to determine which machine learning models would best predict the outcome of future applicants. Libraries we used were scikit-learn v1.0.0 and imablanced-learn

The following models we tested were:

- Native Random Oversampling
```
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```
- SMOTE Oversample
```
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(X_train, y_train)
Counter(y_resampled)
```
- Culster Centroid Undersampling
```
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```
- SMOTEEN Combination Sampling
```
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)
```
- Ensemble Balanced Random Forest Classifier
```
from imblearn.ensemble import BalancedRandomForestClassifier
brfc = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
brfc
```
- Easy Ensemble AdaBoost Classifier
```
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec
```

---

## Results
Results for each category are as follows

- Native Random Oversampling
    
    - Balanced accuracy score: 62%
    
    - Precision score: high risk of 1% and low risk of 100%

    - Recall score: high risk of 67% and low risk of 57%


- SMOTE Oversample
    
    - Balanced accuracy score: 66%
    
    - Precision score: high risk of 1% and low risk of 100%

    - Recall score: high risk of 64% and low risk of 68%

- Culster Centroid Undersampling
    
    - Balanced accuracy score: 54%
    
    - Precision score: high risk of 1% and low risk of 100%

    - Recall score: high risk of 69% and low risk of 40%

- SMOTEEN Combination Sampling
    
    - Balanced accuracy score: 65%
    
    - Precision score: high risk of 1% low risk of 100%

    - Recall score: high risk of 73% and low risk of 57%

- Ensemble Balanced Random Forest Classifier
    
    - Balanced accuracy score: 78%
    
    - Precision score: high risk of 3% and low risk of 100%

    - Recall score: high risk of 70% and low risk of 87%

- Easy Ensemble AdaBoost Classifier
    
    - Balanced accuracy score: 93%
    
    - Precision score: high risk of 9% and low risk of 100%

    - Recall score: high risk of 92% and low risk of 94%

---

## Summary

Looking at the results from all 6 models I would recommend using the Easy Ensemble AdaBoost Classifier to determing which Loan applicants fall into a risk category. Since the accuracy is still off by about 7%, that could be a massive loss in revenue for any false negatives. 

I would recommend using this to not automatically action, but to flag potential high risk individuals. This could then bin those applicants into a risk group, where we could run another model to say whether the applicant fits into an acceptable loss category. Stacking these and applying risk scores that then get fit onto high risk data in a second assessment would allow us to gain more revenue without taking too much loss. i.e. following the model most major companies use to manually review cases for mitigating accuracy.