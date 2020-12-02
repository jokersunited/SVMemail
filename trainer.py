import pandas as pd
import numpy as np
import scipy

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle

from utils import *

# truncate = 2450
# length = 100
#
# email_df = pd.read_csv("spam_or_not_spam.csv", dtype=str)[truncate:truncate+length]
# email_df['email'].dropna(inplace=True)
#
# email_df['email'] = [word_tokenize(str(row).lower()) for row in email_df['email']]
#
# tag_map = word_tag()
#
# for index, text in enumerate(email_df['email']):
#     print(index)
#     clean_words = []
#
#     word_lem = WordNetLemmatizer()
#     for word, tag in pos_tag(text):
#         if word not in stopwords.words('english') and word.isalpha():
#             clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
#             clean_words.append(str(clean_word))
#     print(clean_words)
#     email_df.loc[index+truncate, 'clean_text'] = str(clean_words)
#
#
# email_df.to_pickle("./data.pkl")

email_df = pd.read_pickle("./data.pkl")

#Training
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(email_df['clean_text'], email_df['label'], test_size=0.3)

tfidf_vect = TfidfVectorizer(max_features=4000)
tfidf_vect.fit(email_df['clean_text'])

print(Test_X)

Train_X = tfidf_vect.transform(Train_X)
Test_X = tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X, Train_Y)

predictions_SVM = SVM.predict(Test_X)

filename = "svm_model.pkl"
pickle.dump(SVM, open(filename, 'wb'))

print(predictions_SVM)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)

test = input("\n\nEnter email text here to check")

word_list = word_tokenize(str(test).lower())
print(word_list)
clean_words = []
tag_map = word_tag()
word_lem = WordNetLemmatizer()
for word, tag in pos_tag(word_list):
    if word not in stopwords.words('english') and word.isalpha():
        clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
        clean_words.append(str(clean_word))

test_df = pd.DataFrame(columns=["clean_words"], dtype=object)
test_df.loc[0, 'clean_words'] = str(clean_words)

print(test_df)

test_tfidf = tfidf_vect.transform(test_df['clean_words'])

print(test_tfidf)

prediction = SVM.predict(test_tfidf)
print(prediction)
if prediction:
    print("PHISHING!!!")
else:
    print("CLEAN!!!")



