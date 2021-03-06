
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import *
import sklearn
import pandas as pd
import numpy as np
#import matplotlib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pyplot as plt


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
#print(len(aRevs))


# In[3]:

print(('Shuffled Data'))
daf = shuffle(pd.DataFrame(list(aRevs.items()), columns=["Review", "Class"]), random_state=12)


# In[4]:

print('Data Pickled')
daf.to_pickle("./opDF.pkl")


# In[7]:


x, y = daf["Review"], daf["Class"]


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12 )



# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


# In[11]:


x_train.shape

print(('Vectorized Data'))
# In[12]:


from sklearn.ensemble import AdaBoostClassifier
print(('Imported Adaptive Boost, building classifier'))
clf = AdaBoostClassifier(n_estimators=130, learning_rate=.5)
clf.fit(x_train, y_train)


# In[13]:


clf.score((x_test), y_test)


# In[14]:


ypreds = clf.predict(x_test)


# In[15]:


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
print(f'Average score={((recall+skrecall+skpres+precision+skac+accuracy)/6)*100}%')



tests = open("mixed_test_reviews.txt").readlines()

newTests = []
for rvw in tests:
    rvw = rvw.replace("\n", "")
    newTests.append(rvw)


# In[17]:


preds = (clf.predict(cv.transform(newTests)))


# In[18]:


preds2 = []
for pred in preds:
    pred = round(pred)
    preds2.append(pred)



# In[20]:


with open('res.txt', 'w') as f:
    for prid in preds2:
        f.write("%s\n" % prid)
print(('Outputted to res.txt'))

