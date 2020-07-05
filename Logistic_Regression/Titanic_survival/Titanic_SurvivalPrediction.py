#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

basePath='A:\Git\Logictic_Regression\Titanic_survival'
trainDataFilePath = basePath+'/train.csv'
testDataFilePath = basePath+'/test.csv'

train = pd.read_csv(trainDataFilePath)
test = pd.read_csv(testDataFilePath)


# In[2]:


# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)


# In[3]:


train.head()


# In[4]:


train.count()


# #Check Survived Percentage

# In[5]:


sns.countplot(x='Survived', hue='Pclass', data=train)


# In[6]:


sns.countplot(x='Survived', hue='Sex', data=train)


# In[7]:


#Look at survival rate by sex
train.groupby('Sex')[['Survived']].mean()


# In[8]:


sns.countplot(train['Survived'],label="Count")


# In[9]:


#Handling missing values


# In[10]:


train.isna().sum()


# In[11]:


(train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)


# In[12]:


train.head(10)


# In[13]:


train['Cabin'].unique()


# In[14]:


#Handling missing values for cabin by classifying into numerical value based on categories
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train, test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)


# In[15]:


train.head(10)


# In[16]:


# we can now drop the cabin feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)


# In[17]:


train.head(10)


# In[18]:


#Replace missing age with median
train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())


# In[19]:


#Impute the missing values with most common value

train["Embarked"] = train ["Embarked"].fillna("S")


test["Embarked"] = test["Embarked"].fillna("S")


# In[20]:


train.isna().sum()


# In[21]:


#1.After filling all the missing data now. check correlation matrix using the seaborn heatmap
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(25,15)
plt.show()


# In[22]:


#Now from the above heatmap,we can see that the features are not much correlated. The highest correlation is between SibSp and Parch i.e 0.41.


# In[23]:


train.info()


# In[24]:


#Convert ‘Sex’ feature into numeric.
genders = {"male": 0, "female": 1}
data = [train, test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)


# In[25]:


from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(train[['Embarked']]).toarray())
# merge with main df bridge_df on key values
train = train.join(enc_df)
train


# In[26]:


train.head(10)


# In[27]:


#Converting age to age group
data = [train, test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[28]:


train.head(10)


# In[29]:


#dropping irrelevant features
train = train.drop(["PassengerId","Name","Ticket"], axis=1)
test = test.drop(["PassengerId","Name","Ticket"], axis=1)


# In[30]:


train.head(10)


# In[31]:


#Now all values have numerical equivalent


# In[32]:


#import logistic regression
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score #for accuracy_score
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction


# In[41]:


from sklearn.linear_model import LogisticRegression


Y_target = train["Survived"].values
X_features_one = train[["Pclass", "Sex", "Age", "Fare","Deck"]].values
                               
logistic_model = LogisticRegression()
logistic_model.fit(X_features_one, Y_target)

# Print the Models Coefficients
print(logistic_model.coef_)


# In[44]:


# Make predictions
y_preds = logistic_model.predict(X = X_features_one)

# Generate table of predictions vs actual
pd.crosstab(y_preds,train["Survived"])


# In[46]:


from sklearn.metrics import log_loss
log_loss(y_true=train["Survived"],y_pred=y_preds)


# In[47]:


# Accuracy

logistic_model.score(X = X_features_one ,
                y = train["Survived"])


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(y_true=train["Survived"],y_pred=y_preds)


# In[49]:


from sklearn import metrics 

# View confusion matrix
metrics.confusion_matrix(y_true=train["Survived"],  # True labels
                         y_pred=y_preds) # Predicted labels


# In[51]:


from sklearn import metrics
cm = confusion_matrix(y_true=train["Survived"],y_pred=y_preds)
#visualize confusion matrix 
sns.heatmap(confusion_matrix(train["Survived"],y_preds),annot=True,fmt='3.0f',cmap="summer")
plt.title('cm', y=1.05, size=15)
cm


# In[52]:


# View summary of common classification metrics
print(metrics.classification_report(y_true=train["Survived"],
                              y_pred=y_preds) )


# In[53]:


y_preds_prob = logistic_model.predict_proba(X = X_features_one)


# In[55]:


y_pred_prob_survive = logistic_model.predict_proba(X = X_features_one)[:, 1]
logistic_model.classes_


# In[56]:


fpr, tpr, thresholds = metrics.roc_curve(train["Survived"],
                                         y_pred_prob_survive)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for survival classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[57]:


#Tradeoff between sensivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[58]:


evaluate_threshold(0.5)


# In[60]:


evaluate_threshold(0.4)


# In[ ]:




