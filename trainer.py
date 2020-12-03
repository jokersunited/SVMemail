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

#Training
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(email_df['clean_text'], email_df['label'], test_size=0.3)

tfidf_vect = TfidfVectorizer(max_features=4000)
tfidf_vect.fit(email_df['clean_text'])
# vectorizer = CountVectorizer(email_df['clean_text'])

x = ["Your parcel is ready to be collected, please click this link", "your mother father is fine too", "what the hell you doing sir", "you are a cunt bag"]

Train_X = tfidf_vect.transform(Train_X)
Test_X = tfidf_vect.transform(Test_X)

# plt.spy(Train_X)
# plt.show()
# exit()
#
# # for item in Train_X:
# #     print(item.
# #     exit()
# # print(Train_X[0])
# # print(Test_X.toarray())

tfidf_df = pd.DataFrame(Test_X.todense())
for index, item in tfidf_df.iterrows():
    print(type(item))
    print(item.index)



print(tfidf_df)
# tfidf_df = np.expand_dims(tfidf_df.to_numpy(), 1)
# print(tfidf_df)

print(len(tfidf_df.columns))
print(len(tfidf_df.values[1,:]))
print(len(tfidf_df.values))
plt.scatter(tfidf_df.columns, tfidf_df.values[1,:], c=Test_Y)
plt.show()



print(tfidf_df)
print(Train_X)
print(type(Train_X))


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(Train_X, Train_Y)

predictions_SVM = SVM.predict(Test_X)
print(classification_report(Test_Y, predictions_SVM))
print(cf_64(SVM, Test_X, Test_Y))

filename = "svm_model.pkl"
pickle.dump(SVM, open(filename, 'wb'))

print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)

new_text = convert_text(x, tfidf_vect)

prediction = SVM.predict(new_text)
prediction_prob = SVM.predict_proba(new_text)
prob = [max(x) for x in prediction_prob]
prob_min = [min(x) for x in prediction_prob]

print(prob_min)
print(prob)
print(prediction)



