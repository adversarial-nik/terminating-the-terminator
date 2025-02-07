'''
FWAF - Machine Learning driven Web Application Firewall
Author: Faizan Ahmad
Performance improvements: Timo Mechsner
Website: http://fsecurify.com
https://github.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
import urllib.parse

import matplotlib.pyplot as plt
import pickle

def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

badQueries = loadFile('badqueries.txt')
validQueries = loadFile('goodqueries.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

badCount = len(badQueries)
validCount = len(validQueries)

lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}) # class_weight='balanced')
# lgs = LinearRegression()
lgs.fit(X_train, y_train) #training our model

##############
# Evaluation #
##############

predicted = lgs.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

# save vectorizer and classifier
pickle.dump(vectorizer, open('pickled_vectorizer','wb'))
pickle.dump(lgs, open('pickled_lgs','wb'))

print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  #checking the accuracy
