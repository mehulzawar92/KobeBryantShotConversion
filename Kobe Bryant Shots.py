#!/usr/bin/env python
# coding: utf-8

# # Step 0 - Loading Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


# # Step 1 - Loading datasaet

# In[2]:


dataset = pd.read_csv("Data.csv")


# # Step 2 - Undertanding data

# In[3]:


dataset.head(5)


# In[4]:


dataset.tail(5)


# In[5]:


dataset.shape


# In[6]:


dataset['team_id'].unique()


# In[7]:


# Dropping columns that have zero variance/ insignificant
dataset.drop(['team_id', 'team_name', 'shot_id', 'matchup'], axis = 1, inplace = True)


# In[8]:


dataset.shape


# In[9]:


dataset.columns


# # Step 3 - Visualizing data and gathering insights

# In[10]:


# Checking correlation between variables
plt.figure(figsize = (30,20))
sns.heatmap(data = dataset.corr(), annot = True, cbar = False, cmap = 'rainbow')


# In[11]:


# Game Event ID and Game ID do not provide any useful detail
dataset.drop(['game_event_id', 'game_id'], axis = 1, inplace = True)


# In[12]:


dataset.columns


# In[13]:


# 'Loc x and Lon' and 'Loc y and Lat' are co-related
dataset.drop(['lon', 'lat'], axis = 1, inplace = True)


# In[14]:


dataset.columns


# In[15]:


# Converting minutes and seconds into new variable i.e. time remaining in seconds
dataset['time_remaining'] = dataset['minutes_remaining'] * 60  + dataset['seconds_remaining']


# In[16]:


dataset[['minutes_remaining', 'seconds_remaining', 'time_remaining']]


# In[17]:


# Since, a new variable is featured, existing 2 variables are not required.
dataset.drop(['minutes_remaining', 'seconds_remaining'], axis = 1, inplace = True)


# In[18]:


dataset.shape


# In[19]:


dataset.columns


# In[20]:


sns.countplot(x = 'period', data = dataset)


# In[21]:


sns.countplot(x = 'playoffs', data = dataset)


# In[22]:


# Converting season YYYY-yyyy format to YYYY
dataset['season'] = dataset['season'].apply(lambda x: x[:4])
dataset['season'] = pd.to_numeric(dataset['season'])


# In[23]:


plt.figure(figsize = (10,10))
sns.countplot(x = 'season', data = dataset)


# In[24]:


dataset.columns


# In[25]:


dataset['shot_distance'].hist()


# In[26]:


plt.figure(figsize = (30,20))
sns.heatmap(data = dataset.corr(), annot = True, cbar = False, cmap = 'rainbow')


# In[27]:


dataset.columns


# In[28]:


dataset['shot_type'].unique()


# In[29]:


dataset['shot_zone_area'].unique()


# In[30]:


dataset['shot_zone_basic'].unique()


# In[31]:


dataset['shot_zone_range'].unique()


# In[32]:


sns.pairplot(data = dataset, vars = ['shot_zone_range', 'shot_distance'])


# In[33]:


dataset.drop('shot_zone_range', axis = 1, inplace = True)


# In[34]:


dataset.shape


# In[35]:


dataset.columns


# In[36]:


dataset['game_date'] = pd.to_datetime(dataset['game_date'])


# In[37]:


dataset['game_year'] = dataset['game_date'].dt.year
dataset['game_month'] = dataset['game_date'].dt.month
dataset['game_day'] = dataset['game_date'].dt.dayofweek


# In[38]:


dataset.drop('game_date', axis = 1, inplace = True)


# In[39]:


dataset.columns


# In[40]:


sns.countplot(x = 'game_day', data = dataset)


# In[41]:


sns.heatmap(dataset.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[42]:


dataset.shape


# In[43]:


dataset.head(5)


# In[44]:


data = dataset[dataset['shot_made_flag'].notnull()]


# In[45]:


data.shape


# In[46]:


shot_goal = data[data['shot_made_flag'] == 1]


# In[47]:


shot_not_goal = data[data['shot_made_flag'] == 0]


# In[48]:


sns.heatmap(data.isnull(), yticklabels=False, cbar = False, cmap='Blues')


# In[49]:


print('Total = ', len(data))
print('Number of Goals made = ', len(shot_goal))
print('Number of Goals not made = ', len(shot_not_goal))
print('% of goals made = ', 1 * len(shot_goal)/ len(data) *100)
print('% of goals not made = ', 1 * len(shot_not_goal)/ len(data) *100)


# In[50]:


sns.countplot(x = 'shot_made_flag', data = data)


# In[51]:


sns.countplot(x="period", hue="shot_made_flag", data=data)


# In[52]:


plt.figure(figsize = (15,18))
sns.countplot(y="action_type", hue="shot_made_flag", data=data)


# In[53]:


plt.figure(figsize = (12,18))
sns.countplot(x="season", hue="shot_made_flag", data=data)


# In[54]:


plt.figure(figsize = (12,6))
data['shot_made_flag'].groupby(data['season']).mean().plot()


# In[55]:


plt.figure(figsize=(12,6))
sns.countplot(x="game_day", hue="shot_made_flag", data=data)


# In[56]:


plt.figure(figsize=(12,6))
sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data)


# In[57]:


plt.figure(figsize=(12,16))
sns.countplot(y="opponent", hue="shot_made_flag", data=data)


# In[58]:


plt.figure(figsize=(12,16))
data['shot_made_flag'].groupby(data['opponent']).mean().sort_values().plot(kind = 'barh')


# In[59]:


data.shape


# In[60]:


data.head(5)


# In[61]:


data.columns


# # Step 4 - Making data viable for model

# In[62]:


# Encoding categorical variables
categ = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'opponent', 'period', 'season', 'game_year', 'game_month', 'game_day', 'loc_x', 'loc_y']

for i in categ:
    dummies = pd.get_dummies(data[i], drop_first = True)
    dummies = dummies.add_prefix("{}#".format(i))
    data.drop(i, axis = 1, inplace = True)
    data = data.join(dummies)


# In[63]:


data.shape


# In[64]:


data.head(5)


# In[65]:


X = data.drop('shot_made_flag', axis = 1).values


# In[66]:


y = data['shot_made_flag'].values


# In[67]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Step 5 - Building model

# ## Step 5a - Logistic Regression

# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[72]:


y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


# In[73]:


from sklearn.metrics import confusion_matrix, classification_report


# In[74]:


cm1 = confusion_matrix(y_train, y_pred_train)
cm2 = confusion_matrix(y_test, y_pred_test)


# In[75]:


sns.heatmap(cm1, annot = True, fmt = "d")


# In[76]:


sns.heatmap(cm2, annot = True, fmt = "d")


# In[77]:


print("Accuracy on Train Dataset= ", (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]) * 100, "%")
print("Accuracy on Test Dataset = ", (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]) * 100, "%")


# In[78]:


print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


# # Step 5b - Support Vector Classifier

# In[79]:


from sklearn.svm import SVC


# In[80]:


classifier = SVC(cache_size = 7000, verbose = True)
classifier.fit(X_train, y_train)


# In[81]:


y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


# In[82]:


cm1 = confusion_matrix(y_train, y_pred_train)
cm2 = confusion_matrix(y_test, y_pred_test)


# In[83]:


sns.heatmap(cm1, annot = True, fmt = "d")


# In[84]:


sns.heatmap(cm2, annot = True, fmt = "d")


# In[85]:


print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


# In[86]:


print("Accuracy on Train Dataset= ", (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]) * 100, "%")
print("Accuracy on Test Dataset = ", (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]) * 100, "%")


# ## Step 5c - K-Nearest Neighbours

# In[88]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[89]:


y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


# In[90]:


cm1 = confusion_matrix(y_train, y_pred_train)
cm2 = confusion_matrix(y_test, y_pred_test)


# In[91]:


sns.heatmap(cm1, annot = True, fmt = "d")


# In[92]:


sns.heatmap(cm2, annot = True, fmt = "d")


# In[93]:


print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


# In[94]:


print("Accuracy on Train Dataset= ", (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]) * 100, "%")
print("Accuracy on Test Dataset = ", (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]) * 100, "%")


# ## Step 5d - Random Forest

# In[95]:


from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
randomforest_classifier.fit(X_train, y_train)


# In[96]:


y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


# In[97]:


cm1 = confusion_matrix(y_train, y_pred_train)
cm2 = confusion_matrix(y_test, y_pred_test)


# In[98]:


sns.heatmap(cm1, annot = True, fmt = "d")


# In[99]:


sns.heatmap(cm2, annot = True, fmt = "d")


# In[100]:


print(classification_report(y_train, y_pred_train))
print(classification_report(y_test, y_pred_test))


# In[101]:


print("Accuracy on Train Dataset= ", (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1]) * 100, "%")
print("Accuracy on Test Dataset = ", (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]) * 100, "%")


# ## Summarizing

# In[102]:


table = {'Model' : ['Logistic Regression', 'Supprt Vector Classifier', 'K-Nearest Neighbour', 'Random Forest'], 'Train Accuracy': ['69.33%', '72.35%', '72.73%', '72.73%'], 'Test Accuracy' : ['66.30%', '67.70%', '57.69%', '57.69%']}
summary_table = pd.DataFrame(table, columns = ['Model', 'Train Accuracy', 'Test Accuracy'])
print(summary_table)


# In[ ]:




