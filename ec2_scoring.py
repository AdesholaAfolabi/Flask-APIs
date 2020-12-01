#scoring from the server. First CTR (2019)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport

from sklearn.metrics import precision_recall_curve, make_scorer, confusion_matrix, classification_report
from sklearn.externals import joblib

import itertools

w = pd.DataFrame()
c = 0

for data in pd.read_csv("/var/www/5bfc4eca-4c1d-4536-8bdf-880aaea4f516.csv", chunksize=600000):
    msisdn = []
    data.dropna(subset=['msisdn'], axis=0, inplace=True)
    data.drop_duplicates('msisdn', inplace=True)
    for number in list(data.msisdn):
        msisdn.append(number)

    list_region = ["South West", "South East", "South South", "North Central", "North West", "North East"]
    #list_os_name = ["android", "proprietary os", "symbian os", "ios", "series30", "series40", "samsung os", "blackberry os", "mtk rtos", "windows phone", "nucleus", "series 60", "lg os", "nokia os", "asha software", "windows mobile", "siemens os", "spreadtrum os", "belle", "yun os", "isa", "mocor os", "tizen", "panasonic os", "motomagx", "bada"]
    features = ["age", "location_region", "customer_class", "customer_value", "spend_total"]
    data = data[features]
    data['location_region'] = data['location_region'].replace(to_replace = np.nan, value ="South West")
    data['location_region']= data['location_region'].apply(lambda x : x if x in list_region else "South West")
    #data['os_name'] = data['os_name'].replace(to_replace = np.nan, value ="android")
    data['age'] = abs(data.age)
    data['age'] = data['age'].apply(lambda x : 60.0 if x >= 60.0 else x)
    data['age'].fillna((data['age'].mean()), inplace=True)
    #categorical_variable = ["location_region", "os_name"]
    data['customer_value'] = data.customer_value.map({'top':5, 'very high':4, 'high':3, 'medium':2, 'low':1})
    data['customer_value'].fillna((data['customer_value'].mean()), inplace=True)
    data['spend_total'].fillna((data['spend_total'].mean()), inplace=True)
    data['age'].fillna((data['age'].mean()), inplace=True)
    data['customer_class'].fillna((data['customer_class'].mean()), inplace=True)

    model_data = pd.get_dummies(data)
    model = joblib.load("general.pkl")
    result = model.predict_proba(model_data)[:,1]
    label = dict(zip(msisdn, result))
    pred = pd.DataFrame.from_dict(label, orient = 'index', columns=['predictions'])
    w = pd.concat([w, pred])
    c +=1
    os.system("sudo sysctl vm.drop_caches=3")
    print(c)
w = w.reset_index()
w.to_csv("scored_bib.csv")
