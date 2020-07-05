#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

basePath='A:\Git\Logistic_Regression\Titanic_survival'
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


#handling outliers
# Detect outliers in the continuous columns

cols = list(train)
outliers = pd.DataFrame(columns=['Feature', 'Number of Outliers'])

for column in cols:  # Iterate through each feature
    if column in train.select_dtypes(include=np.number).columns:
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - (1.5 * iqr)
        fence_high = q3 + (1.5 * iqr)

        # finding the number of outliers using 'and(|) condition.
        total_outlier = train[(train[column] < fence_low) |
                                  (train[column] > fence_high)].shape[0]
        outliers = outliers.append(
            {
                'Feature': column,
                'Number of Outliers': total_outlier
            },
            ignore_index=True)
outliers


# In[10]:


#Handling missing values


# In[11]:


train.isna().sum()


# In[12]:


(train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)


# In[13]:


train.head(10)


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


train.head()


# In[22]:


#1.After filling all the missing data now. check correlation matrix using the seaborn heatmap
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(25,15)
plt.show()


# In[23]:


train.info()


# In[24]:


train["Sex"].unique()


# In[25]:


#Convert the categorical variables into integers

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1


# In[26]:


train["Sex"].unique()


# In[27]:


from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(train[['Embarked']]).toarray(),columns=['S', 'C', 'Q'])
# merge with main df bridge_df on key values
train = train.join(enc_df)
train


# In[28]:


train.isna().sum()


# In[29]:


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


# In[30]:


train.head(10)


# In[31]:


#dropping irrelevant features
train = train.drop(["PassengerId","Name","Ticket"], axis=1)
test = test.drop(["PassengerId","Name","Ticket"], axis=1)


# In[38]:


train = train.drop(["Embarked"], axis=1)
test = test.drop(["Embarked"], axis=1)


# In[39]:


#Scaling the numerical data:
from sklearn.preprocessing import StandardScaler
#train_numerical_features = list(train.select_dtypes(include=['int64', 'float64', 'int32']).columns)
ss_scaler = StandardScaler()
#train_df_ss = pd.DataFrame(data = train)
train[[ "Fare"]] = ss_scaler.fit_transform(train[["Fare"]])


# In[40]:


train.head(10)


# In[41]:


#Now all values have numerical equivalent
train.iloc[:,1:11]
train.iloc[:,0:1]


# In[45]:


#selecting best features
X = train.iloc[:,1:11]  #independent columns
y = train.iloc[:,0:1]    #target column i.e price range
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 8)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print('Selected features: %s' % list(X.columns[rfe.support_]))
print("Feature Ranking: %s" % (fit.ranking_))


# In[46]:


#import logistic regression
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score #for accuracy_score
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction


# In[47]:


from sklearn.linear_model import LogisticRegression


Y_target = train["Survived"].values
X_features_one = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'S', 'C', 'Q']].values
                               
logistic_model = LogisticRegression()
logistic_model.fit(X_features_one, Y_target)

# Print the Models Coefficients
print(logistic_model.coef_)
logistic_model.score(X_features_one, Y_target)


# In[48]:


# Make predictions
y_preds = logistic_model.predict(X = X_features_one)

# Generate table of predictions vs actual
pd.crosstab(y_preds,train["Survived"])


# In[49]:


from sklearn.metrics import log_loss
log_loss(y_true=train["Survived"],y_pred=y_preds)


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(y_true=train["Survived"],y_pred=y_preds)


# In[52]:


from sklearn import metrics 

# View confusion matrix
metrics.confusion_matrix(y_true=train["Survived"],  # True labels
                         y_pred=y_preds) # Predicted labels


# In[53]:


from sklearn import metrics
cm = confusion_matrix(y_true=train["Survived"],y_pred=y_preds)
#visualize confusion matrix 
sns.heatmap(confusion_matrix(train["Survived"],y_preds),annot=True,fmt='3.0f',cmap="summer")
plt.title('cm', y=1.05, size=15)
cm


# In[54]:


# View summary of common classification metrics
print(metrics.classification_report(y_true=train["Survived"],
                              y_pred=y_preds) )


# # Precision Recall Curve

# In[59]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = logistic_model.predict_proba(X = X_features_one)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(train["Survived"], y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# Above you can clearly see that the recall is falling of rapidly at a precision of around 80%. 
# Because of that we may want to select the precision/recall tradeoff before that — maybe at around 75 %.
# By looking at the plots we need a threshold of around 0.4

# Another way to evaluate and compare your binary classifier is provided by the ROC AUC Curve. 
# This curve plots the true positive rate (also called recall) against the false positive rate (ratio of incorrectly classified negative instances)

# In[60]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(train["Survived"], y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.

# In[61]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(train["Survived"], y_scores)
print("ROC-AUC-Score:", r_a_score)


# In[ ]:




