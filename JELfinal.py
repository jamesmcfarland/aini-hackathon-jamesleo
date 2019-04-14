#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import matplotlib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# In[2]:


fakes = open("fake_reviews.txt").readlines()
reals = open("real_reviews.txt").readlines()


fakes2 = []
for rev in  fakes:
    rev = rev.replace('\n', '')
    fakes2.append(rev)

reals2 = []
for tv in  reals:
    tv= tv.replace('\n', '')
    reals2.append(tv)


fakeD = {key: 0 for (key) in fakes2}
realD = {key: 1 for (key) in reals2}

aRevs = {**fakeD, **realD}
print(('Read Data'))


# In[3]:


daf = shuffle(pd.DataFrame(list(aRevs.items()), columns=["Review", "Class"]), random_state=12)
print(('Shuffled Data'))


# In[4]:


daf.to_pickle("./opDF.pkl")
print(('Pickled Data'))


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
y = daf["Class"]
cv = TfidfVectorizer()
x_vect = cv.fit_transform(daf["Review"])
x_vect.shape
print(('Vectorized Data'))

# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x_vect,y, test_size=0.2, random_state=12 )


# In[10]:


from sklearn.ensemble import AdaBoostClassifier
print(('Imported ADABoost, building classifier'))
clf = AdaBoostClassifier(n_estimators=130, learning_rate=.5).fit(x_train, y_train)


# In[11]:


clf.score(x_test, y_test)


# In[12]:


ypreds = clf.predict(x_test)
print("Y predictions made")
# In[13]:


tpos, tneg, fpos, fneg = 0, 0, 0, 0

for prediction, correct_value in zip(ypreds, y_test):
    if prediction == 1 and correct_value == 1:
        tpos += 1
    if prediction == 1 and correct_value == 0:
        fpos += 1
    if prediction == 0 and correct_value == 0:
        tneg += 1
    if prediction == 0 and correct_value == 1:
        fneg += 1
        
        
        
recall = (tpos) / (tpos + fneg)
skrecall = sklearn.metrics.recall_score(y_test, ypreds)
skpres = sklearn.metrics.precision_score(y_test, ypreds)
skac = clf.score(x_test, y_test)
print(f'Recall: {recall:.2f}')
print(f'Sklearn recall: {sklearn.metrics.recall_score(y_test, ypreds):.2f}')
precision = (tpos) / (tpos + fpos)
print(f'Precision: {precision:.2f}')
print(f'Skearn precision: {sklearn.metrics.precision_score(y_test, ypreds):.2f}')
accuracy = (tpos + tneg) / (tpos + tneg + fpos + fneg)
print(f'Accuracy: {accuracy:.2f}')
print(f'Sklearn accuracy: {clf.score(x_test, y_test):.2f}')
print(f'Average score={round(((recall+skrecall+skpres+precision+skac+accuracy)/6)*100)}%')


# In[14]:


tests = open("mixed_test_reviews.txt").readlines()

newTests = []
for rvw in tests:
    rvw = rvw.replace("\n", "")
    newTests.append(rvw)
print(('Test Data Sanitized'))

# In[15]:

print("Predicting new")
preds = (clf.predict(cv.transform(newTests)))


# In[16]:


preds2 = []
for pred in preds:
    pred = round(pred)
    preds2.append(pred)

# In[19]:


with open('res.txt', 'w') as f:
    for prid in preds2:
        f.write("%s\n" % prid)
print(('Outputted to res.txt'))

