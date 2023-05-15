import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("tab10")
from scipy import stats
from sklearn.model_selection import train_test_split
import os
from langdetect import detect
import unicodedata
import re
import nltk
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
seed = 1349


def detect_language(text):
    try:
        return detect(text)
    except:
        return None
    
def acquire():
    '''
    Obtains the vanilla version of both the red and white wine dataframe
    INPUT:
    NONE
    OUTPUT:
    red = pandas dataframe with red wine data
    white = pandas dataframe with white wine data
    '''
    r = pd.read_csv('langr_raw.csv')
    python = pd.read_csv('langp_raw.csv')
    return r, python

def prepare_mvp():
    '''
    Takes in the vanilla red and white wine dataframes and returns a cleaned version that is ready 
    for exploration and further analysis
    INPUT:
    NONE
    OUTPUT:
    wines = pandas dataframe with both red and white wine prepped for exploration
    '''
    r, python = acquire()
    df = pd.concat([r, python], ignore_index = True)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop_duplicates()
    df['readme_language'] = df['readme_contents'].apply(detect_language)
    df = df[df['readme_language'] == 'en']
    df = df.drop('readme_language', axis=1)
    return df

def basic_clean(string):
    '''
    takes in a string and outputs a basic-cleaned version:
                    -lowercase
                    -normalized to unicode set
                    -replaced non-word and non-singlespace,non-singlequote chars with ''
    '''
    lowered = string.lower()
    normalized = unicodedata.normalize('NFKD', lowered).encode('ascii','ignore').decode('utf-8')
    basic_clean = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
    return basic_clean

def tokenize(string):
    '''
    takes in a string and outputs a tokenized version:
    
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    tokenized = tokenizer.tokenize(string, return_str=True)
    return tokenized

def remove_stopwords(some_string, extra_words=['r','be', 'python'], keep_words=[]):
    '''
    remove stopwords will take in a single document as a string
    and return a new string that has stopwords removed
    '''
    stopwords_custom = set(stopwords.words('english')) - \
    set(keep_words)
    stopwords_custom = list(stopwords_custom.union(extra_words))
    return ' '.join([word for word in some_string.split()
                     if word not in stopwords_custom])

def lemmatize(some_string):
    '''
    lemmatize will take in the contents of a single string,
    split up the contents with split()
    use the split contents as a list to apply a lemmatizer to
    each word,
    and return a single string of the lemmatized words joined
    with a single instance of whitespace (' '.join())
    '''
    lemmatizer = nltk.WordNetLemmatizer()
    return ' '.join(
        [lemmatizer.lemmatize(word,'v'
                             ) for word in some_string.split()])

def final_wrangle(df):
    df['cleaned'] = df.readme_contents.apply(basic_clean)
    df['tokenized'] = df.cleaned.apply(tokenize)
    df['lemmatized'] = df.tokenized.apply(lemmatize)
    df['stopped'] = df.lemmatized.apply(remove_stopwords)
    df =df.drop(columns=['cleaned','tokenized','lemmatized'])
    X = df.stopped
    y = df.language
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                 test_size=0.2, 
                 random_state=1349)
    X_train, X_validate = train_test_split(X_train,
                                   train_size=0.7,
                                   random_state=1349)
    y_train, y_validate = train_test_split(y_train,
                                   train_size=0.7,
                                   random_state=1349)
    return X_train, X_validate, X_test, y_train, y_validate, y_test


