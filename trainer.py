import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

from utils import *

email_df = pd.read_csv("spam_or_not_spam.csv")[:30]
email_df['email'].dropna(inplace=True)

email_df['email'] = [word_tokenize(str(row).lower()) for row in email_df['email']]

tag_map = word_tag()

for index, text in enumerate(email_df['email']):
    print(index)
    clean_words = []

    word_lem = WordNetLemmatizer()
    for word, tag in pos_tag(text):
        if word not in stopwords.words('english') and word.isalpha():
            clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
            clean_words.append(clean_word)
    email_df.loc[index, 'clean_text'] = str(clean_words)


email_df.to_pickle("./data.pkl")

#Training
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(email_df['clean_text'],email_df['label'],test_size=0.2)

tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(email_df['clean_text'])

Train_X = tfidf_vect.transform(Train_X)
Test_X = tfidf_vect.transform(Test_X)

print(tfidf_vect.vocabulary)

print(Train_X)
print(Test_X)

