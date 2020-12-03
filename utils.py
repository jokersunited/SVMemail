from collections import defaultdict
from nltk.corpus import wordnet as wn

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
import io
import base64

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


def convert_text(text_list, tfidf):
    clean_df = pd.DataFrame(columns=["clean_words"], dtype=object)

    for index, text in enumerate(text_list):
        print("\n[*] Entry " + str(index))
        print(text)
        word_list = word_tokenize(text.lower())

        clean_words = []
        tag_map = word_tag()
        word_lem = WordNetLemmatizer()
        for word, tag in pos_tag(word_list):
            if word not in stopwords.words('english') and word.isalpha():
                clean_word = word_lem.lemmatize(word, tag_map[tag[0]])
                clean_words.append(str(clean_word))


        clean_df.loc[index, 'clean_words'] = str(clean_words)
        df_tfidf = tfidf.transform(clean_df['clean_words'])

    return df_tfidf


def cf_64(svm, test_x, test_y):
    plot_confusion_matrix(svm, test_x, test_y)
    plt.savefig("help.png")
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_enc = base64.b64encode(pic_IObytes.read())
    return pic_enc


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs