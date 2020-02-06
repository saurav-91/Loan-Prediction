# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:38:57 2020

@author: SAURAV COOL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('credit_train.csv')
test = pd.read_csv('credit_test.csv')

train_original = train.copy()
test_original = test.copy()
train.columns
test.columns

print(train.dtypes)
print('train data shape:',train.shape)
print(train.head())
print('test data shape:',test.shape)
print(test.head())

print(train['Class'].value_counts(normalize =True)*100)
print(train['Class'].value_counts(normalize= True).plot.bar(title= 'Class'))
print(train['Class'].value_counts())

print(train['Amount'].value_counts(normalize = True)*100)
print(train['Amount'].value_counts(normalize = True).plot.bar(title = 'Amount'))
print(train['Amount'].value_counts)


#finding correlations amoung datset
pd.set_option('display.width',100)
pd.set_option('precision',3)
correlations = train.corr(method='pearson')
print(correlations)
sns.heatmap(correlations)

#summary of the trainset
print(train.describe())

plt.figure(1)
plt.subplot(121)
sns.distplot(train["Amount"])

#Finding missing data
print(train.isnull().values.any())
print(train.isnull().sum().sort_values(ascending= False))
#there is no missing data

print(train['Class'].value_counts())
print(train['Class'].value_counts(normalize= True).plot.bar(title= 'Class')) 
print(train['Class'].value_counts(normalize =True)*100)
#in trainset about 99.82% are not fraud and 0.171% are fraud transcation


#Transcation Amount
(ax1, ax2) = plt.subplots(ncols=2,figsize=(12,6))
s=sns.boxplot(ax = ax1, x = 'Class',y='Amount',hue = 'Class', data = train, palette='PRGn',showfliers= True)
s=sns.boxplot(ax = ax2, x = 'Class',y='Amount',hue = 'Class', data = train, palette='PRGn',showfliers= False)
plt.show()


tmp = train[['Amount', 'Class']]
class_0 = tmp.loc[train['Class']==0]['Amount']
class_1 = tmp.loc[train['Class']==1]['Amount']
print(class_0.describe())

# the real transcation have a larger mean value larger Q1and smaller Q3 and Q4 and larger outliers
# the fraud transcation have smaller Q1 and mean and larger Q3 and Q4 and ssmaller outlier

# from the direct correlated values (V20, Amount) and(V7, Amount) 

s = sns.lmplot( x='V20', y ='Amount', data = train,hue='Class', fit_reg= True,scatter_kws={'s':2})
s = sns.lmplot( x='V7', y='Amount', data = train,hue='Class',fit_reg= True,scatter_kws={'s':2})
plt.show()
# the regression line for class=0 have a positive slope
# the regression line for class=1 have a smaller positive slope

# let plot invesre correlated plot 
s = sns.lmplot(x='V2', y='Amount',data=train,hue='Class',fit_reg =True,scatter_kws={'s':2})
s = sns.lmplot(x='V5', y='Amount',data = train, hue= 'Class',fit_reg = True,scatter_kws={'s':2})
plt.show()

# the regression line for Class=0 have a negative slope
# the regression line for Class=1 have a very small negative slope

from sklearn.preprocessing import StandardScaler
train['Amount']= StandardScaler().fit_transform(train[['Amount']])
train.head()

#Model Building
# drop class column 

#data divided into train and validation part
x= train.drop("Class",1)
y= train[["Class"]]
test_2 = test.drop("Class",1)
test_y = test[["Class"]]


#splitting the data
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size=0.3, random_state=1)


#applying logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic_model = LogisticRegression(random_state=1)

#fit the model
logistic_model.fit(x_train,y_train)
pred_cv_logistic=logistic_model.predict(x_cv)
df = pd.DataFrame(pred_cv_logistic)

score_logistic =accuracy_score(pred_cv_logistic,y_cv)*100
print(score_logistic)
pred_test_logistic = logistic_model.predict(test_2)
d_test = pd.DataFrame(pred_test_logistic)
print(pred_test_logistic)
score_logistic_test= accuracy_score(pred_test_logistic,test_y)*100
print(score_logistic_test)




#calculating confusion matrix
from sklearn.metrics import  classification_report,confusion_matrix
print(confusion_matrix(y_cv,pred_cv_logistic))
print(classification_report(y_cv,pred_cv_logistic))

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_y,pred_test_logistic))
print(classification_report(test_y,pred_test_logistic))



#importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state = 1)
#Fit the model
tree_model.fit(x_train,y_train)
pred_cv_tree = tree_model.predict(x_cv)
score_tree = accuracy_score(pred_cv_tree, y_cv)*100
print(score_tree)
pred_test_tree = tree_model.predict(test_2)
decision_test = pd.DataFrame(pred_test_tree)
print(pred_test_tree)
score_tree_test= accuracy_score(pred_test_tree,test_y)*100
print(score_tree_test)

# calculating confusion matrix
from sklearn.metrics import  classification_report,confusion_matrix
print(confusion_matrix(y_cv,pred_cv_tree))
print(classification_report(y_cv,pred_cv_tree))

#import Support Vector Machine Classifier
from sklearn import svm
clf = svm.SVC(kernel='linear')
# train models using training set
clf.fit(x_train,y_train)
pred_cv_svm= clf.predict(x_cv)
score_svm = accuracy_score(pred_cv_svm,y_cv)
print(score_svm)
pred_test_svm = clf.predict(test_2)
svm_test = pd.DataFrame(pred_test_svm)
print(pred_test_svm)
score_svm_test=accuracy_score(pred_test_svm,test_y)
print(score_svm_test)

# calculating confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_cv,pred_cv_svm))
print(classification_report(y_cv,pred_cv_svm))

from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=1,max_depth=10,n_estimators=10)
forest_model.fit(x_train,y_train)
pred_cv_forest = forest_model.predict(x_cv)
score_forest = accuracy_score(pred_cv_forest,y_cv)*100
print(score_forest)
pred_test_forest = forest_model.predict(test_2)
print(pred_test_forest)
random_test = pd.DataFrame(pred_test_forest)
score_forest_test = accuracy_score(random_test,test_y)*100
print(score_forest_test)

# calculating confusion matrix
from sklearn.metrics import  classification_report,confusion_matrix
print(confusion_matrix(y_cv,pred_cv_forest))
print(classification_report(y_cv,pred_cv_forest))


# on applying K-fold Cross Validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(clf,x_cv,y_cv,cv= 10,scoring = 'accuracy').mean())
print(cross_val_score(clf,test_2,test_y,cv= 10,scoring = 'accuracy').mean())
















