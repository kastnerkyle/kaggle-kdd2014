# Author: Kyle Kastner
# Extended from "beating the benchmark" code by Abhishek Thakur
# License: BSD 3 Clause
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import check_arrays
from scipy import sparse
import re

print("Reading and cleaning data.")
DATA_DIR = "donorschoose/"
projects = pd.read_csv(DATA_DIR + 'projects.csv')
outcomes = pd.read_csv(DATA_DIR + 'outcomes.csv')
essays = pd.read_csv(DATA_DIR + 'essays.csv')

projects = projects.sort('projectid')
outcomes = outcomes.sort('projectid')
essays = essays.sort('projectid')
ess_proj = pd.merge(essays, projects, on='projectid')
outcomes_arr = np.array(outcomes)

labels = outcomes_arr[:, 1]

columns = list(ess_proj.columns)
train_idx = np.where(ess_proj['date_posted'] < '2014-01-01')[0]
test_idx = np.where(ess_proj['date_posted'] >= '2014-01-01')[0]


def clean(s):
    return " ".join(
        re.findall(
            r'\w+', s, flags=re.UNICODE | re.LOCALE)).lower()


def count(s):
    return len(s.split(" "))


def tfidf(name):
    tfidf = TfidfVectorizer(min_df=3, max_features=1000, stop_words='english')
    d = ess_proj[name].fillna("garbage").apply(clean)
    tfidf.fit(d[train_idx])
    e = tfidf.transform(d)
    return e


def count_words(name):
    e = np.array(ess_proj[name].fillna("garbage").apply(clean).apply(count))
    return e


def label_encoder(name, filltype='string'):
    print("Encoding label for %s" % name)
    le = LabelEncoder()
    if filltype == 'string':
        d = ess_proj[name].fillna("garbage")
    elif filltype == 'int':
        d = ess_proj[name].fillna(1999)
    le.fit(d[train_idx])
    return le.transform(d)

print("Extracting features.")
rules = {
    'title': tfidf('title'),
    'need_statement': tfidf('need_statement'),
    'essay': tfidf('essay'),
    }

all_tr = []
all_ts = []
feature_names = []
for k in rules.keys():
    print("Fetching features for %s" % k)
    r = rules[k]
    check_arrays(r)
    if len(r.shape) < 2:
        tr = r[train_idx][:, np.newaxis]
        ts = r[test_idx][:, np.newaxis]
    else:
        tr = r[train_idx]
        ts = r[test_idx]
    all_tr.append(tr)
    all_ts.append(ts)
    feature_names.append(k)

X_train = sparse.hstack(all_tr)
X_test = sparse.hstack(all_ts)
y = np.array(labels == 't').astype(int)

clf = MultinomialNB()
print("Finding support.")
sel = RFECV(clf, step=.01, cv=5, scoring='roc_auc')
sel.fit(X_train, y)
print("Number of support samples = %i" % sel.n_features_)
print("Training.")
clf.fit(X_train.tocsc()[:, sel.support_], y)
preds = clf.predict_proba(X_test.tocsc()[:, sel.support_])[:, 1]

print("Writing predictions.")
sample = pd.read_csv(DATA_DIR + 'sampleSubmission.csv')
sample = sample.sort('projectid')
sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index=False)
