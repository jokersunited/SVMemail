import pandas as pd
import numpy as np
import scipy

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import PCA
from sklearn import svm

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
import pickle

from utils import *

# load_dataset("spam_or_not_spam.csv", "data.pkl")

email_df = pd.read_pickle("./data.pkl")

#=========== Training ==============
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(email_df['clean_text'], email_df['label'], test_size=0.3)

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english', max_features=4000)
tfidf_vect.fit(email_df['clean_text'])

Train_X = tfidf_vect.transform(Train_X)
Test_X = tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(Train_X, Train_Y)

predictions_SVM = SVM.predict(Test_X)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)
#=========== Training ==============

#=========== Saving model ===========

accuracy = "SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100
report = classification_report(Test_Y, predictions_SVM)
cfm = cf_64(SVM, Test_X, Test_Y))

filename = "svm_model.pkl"
pickle.dump({'model': SVM, 'report': report, 'cfm': cfm, 'accuracy': accuracy}, open(filename, 'wb'))

#=========== Saving model ===========


#
# new_text = convert_text(x, tfidf_vect)
#
# prediction = SVM.predict(new_text)
# prediction_prob = SVM.predict_proba(new_text)
# prob = [max(x) for x in prediction_prob]
# prob_min = [min(x) for x in prediction_prob]
#
# print(prob_min)
# print(prob)
# print(prediction)



