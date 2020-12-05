from collections import defaultdict
from nltk.corpus import wordnet as wn

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
import time

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

from sklearn import model_selection, svm

from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
import io
import base64
import pickle

import matplotlib.pyplot as plt


def word_tag():
    tag_map = defaultdict(lambda: wn.NOUN)

    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    return tag_map


def load_dataset(filepath, output, truncate=False):
    if not truncate:
        email_df = pd.read_csv(filepath, dtype=str)
    else:
        email_df = pd.read_csv(filepath, dtype=str)[truncate[0]:truncate[1]]
    email_df['email'].dropna(inplace=True)
    email_df['email'] = [word_tokenize(str(row)[9:].lower()) for row in email_df['email']]

    tag_map = word_tag()

    for index, text in enumerate(email_df['email']):
        print("[+] Processing row " + str(index) + "!")
        clean_words = []

        word_lem = WordNetLemmatizer()
        for word, tag in pos_tag(text):
            if word.isdigit():
                word = "number"
            if word not in stopwords.words('english') and word.isalpha():
                clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
                clean_words.append(str(clean_word))
        if not truncate:
            email_df.loc[index, 'clean_text'] = str(clean_words)
        else:
            email_df.loc[index+truncate, 'clean_text'] = str(clean_words)

    email_df.to_pickle(output)


def convert_text(text_list, label):
    clean_df = pd.DataFrame(columns=["clean_text"], dtype=object)

    for index, text in enumerate(text_list):
        print("\n[*] Entry " + str(index))
        word_list = word_tokenize(text.lower())
        print(word_list)

        clean_words = []
        tag_map = word_tag()
        word_lem = WordNetLemmatizer()
        for word, tag in pos_tag(word_list):
            if word not in stopwords.words('english') and word.isalpha():
                clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
                clean_words.append(str(clean_word))

        print(clean_words)
        clean_df.loc[index, 'clean_text'] = str(clean_words)
        clean_df.loc[index, 'label'] = label[index]

    return clean_df


def cf_64(svm, test_x, test_y):
    plot_confusion_matrix(svm, test_x, test_y)
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_enc = base64.b64encode(pic_IObytes.read())
    return pic_enc

def retrain_model(email_df, size):
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(email_df['clean_text'], email_df['label'].astype(int), test_size=size)

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
    print(report)
    # cfm = cf_64(SVM, Test_X, Test_Y)

    filename = "svm_model_" + str(time.time()) + ".pkl"
    # pickle.dump(email_df, open('data.pkl', 'wb'))
    pickle.dump({'dataset': email_df, 'model': SVM, 'report': report, 'accuracy': accuracy, 'vector': tfidf_vect, 'training': size}, open(filename, 'wb'))
    return filename