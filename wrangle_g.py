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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#CATboost imports
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
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

    train_validate, test = train_test_split(df,
                                        random_state=1349,
                                        train_size=0.8,
                                       stratify=df.language)
    train, validate = train_test_split(train_validate,
                                   random_state=1349,
                                   train_size=0.7,
                                  stratify=train_validate.language)
    return train, validate, test

def stat():
    df = prepare_mvp()
    train, validate, test =final_wrangle(df)
    α = 0.05
    train['token_cnt'] = [len(row['stopped'].split()) for _, row in train.iterrows()]
    r_token_values = train[train.language == 'R'].token_cnt
    python_values = train[train.language == 'Python'].token_cnt
    t, p = stats.ttest_ind(r_token_values,python_values,equal_var=True)
    print(t,p,α)
    print(f'p = {p:.4f}')

ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']

def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]




def explore():
    df = prepare_mvp()
    train, validate, test = final_wrangle(df)
    r_df = clean(' '.join(train[train.language=='R']['stopped']))
    python_df = clean(' '.join(train[train.language=='Python']['stopped']))
    all_df = clean(' '.join(train['stopped']))
    r_freq = pd.Series(r_df).value_counts()
    python_freq = pd.Series(python_df).value_counts()
    all_freq = pd.Series(all_df).value_counts()
    word_counts = pd.concat([r_freq, python_freq,all_freq], axis=1
            ).fillna(0
                    ).astype(int)
    word_counts.columns = ['r', 'python' ,'all']
    word_counts.head()
    return r_df, python_df, all_df, r_freq, python_freq, all_freq, word_counts


def exp1():
    df = prepare_mvp()
    train, validate, test = final_wrangle(df)
    r_df = clean(' '.join(train[train.language=='R']['stopped']))
    python_df = clean(' '.join(train[train.language=='Python']['stopped']))
    all_df = clean(' '.join(train['stopped']))
    r_freq = pd.Series(r_df).value_counts()
    python_freq = pd.Series(python_df).value_counts()
    all_freq = pd.Series(all_df).value_counts()
    word_counts = pd.concat([r_freq, python_freq,all_freq], axis=1
         ).fillna(0
                 ).astype(int)
    word_counts.sort_values('all', ascending=False
                       )[['r','python']].head(20).plot.barh()            

train, validate, test = final_wrangle(prepare_mvp())
train['token_cnt'] = [len(row['stopped'].split()) for _, row in train.iterrows()]
train.rename(columns = {'language':'language_R'}, inplace = True)
validate.rename(columns = {'language':'language_R'}, inplace = True)
test.rename(columns = {'language':'language_R'}, inplace = True)
train = train.drop(columns=['repo','readme_contents'])
validate = validate.drop(columns=['repo','readme_contents'])
test = test.drop(columns=['repo','readme_contents'])
# split train into X (dataframe, drop target) & y (series, keep target only)
X_train = train['stopped']
y_train = train['language_R']
# split validate into X (dataframe, drop target) & y (series, keep target only)
X_validate = validate['stopped']
y_validate = validate['language_R']

# split test into X (dataframe, drop target) & y (series, keep target only)
X_test = test['stopped']
y_test = test['language_R']

tfidf = TfidfVectorizer()
X_bow = tfidf.fit_transform(X_train)






def dtc_model():
    dtc = DecisionTreeClassifier(max_depth=4)
    # fit the model to the TRAIN dataset:
    dtc.fit(X_bow, y_train)
    dtc.score(X_bow,y_train)
    dtc_preds = dtc.predict(X_bow)
    pd.crosstab(dtc_preds,y_train)
    # as with any other sklearn transformation, 
    # transform only on our validate and/or test, 
    # only fit on train
    X_validate_bow = tfidf.transform(X_validate)
    dtc.score(X_validate_bow, y_validate)
    print(f'Accuracy-Train {round(dtc.score(X_bow,y_train),4)}')
    print(f'Accuracy-Validate {round(dtc.score(X_validate_bow,y_validate),4)}')
    print(classification_report(y_train,dtc_preds))
    print(classification_report(y_validate,dtc.predict(X_validate_bow)))

def rfc():
    rf6 = RandomForestClassifier(n_estimators=201,max_depth=2,min_samples_leaf=2)
    rf6.fit(X_bow, y_train)
    rf6.score(X_bow, y_train)
    rf6_preds = rf6.predict(X_bow)
    pd.crosstab(rf6_preds,y_train)
    X_validate_bow = tfidf.transform(X_validate)
    rf6.score(X_validate_bow, y_validate)
    print(f'Accuracy-Train {round(rf6.score(X_bow,y_train),4)}')
    print(f'Accuracy-Validate {round(rf6.score(X_validate_bow,y_validate),4)}')
    print(classification_report(y_train,rf6_preds))
    print(classification_report(y_validate,rf6.predict(X_validate_bow)))

def gdtc():
    # as with any other sklearn transformation, 
    # transform only on our validate and/or test, 
    # only fit on train
    dtc = DecisionTreeClassifier(max_depth=6)
    # fit the model to the TRAIN dataset:
    dtc.fit(X_bow, y_train)
    dtc.score(X_bow,y_train)
    dtc_preds = dtc.predict(X_bow)
    pd.crosstab(dtc_preds,y_train)
    # as with any other sklearn transformation, 
    # transform only on our validate and/or test, 
    # only fit on train
    X_validate_bow = tfidf.transform(X_validate)
    dtc.score(X_validate_bow, y_validate)
    print(f'Accuracy-Train {round(dtc.score(X_bow,y_train),4)}')
    print(f'Accuracy-Validate {round(dtc.score(X_validate_bow,y_validate),4)}')
    print(classification_report(y_train,dtc_preds))
    print(classification_report(y_validate,dtc.predict(X_validate_bow)))

def bgram():
    tfidf2 = TfidfVectorizer(ngram_range=(2,2))
    X_bow = tfidf2.fit_transform(X_train)
    dtc = DecisionTreeClassifier(max_depth=4)
    dtc.fit(X_bow, y_train)
    dtc.score(X_bow, y_train)
    dtc_preds = dtc.predict(X_bow)
    pd.crosstab(dtc_preds,y_train)
    X_validate_bow = tfidf2.transform(X_validate)
    dtc.score(X_validate_bow, y_validate)
    print(f'Accuracy-Train {round(dtc.score(X_bow,y_train),4)}')
    print(f'Accuracy-Validate {round(dtc.score(X_validate_bow,y_validate),4)}')
    print(classification_report(y_train,dtc_preds))
    print(classification_report(y_validate,dtc.predict(X_validate_bow)))


def prediction_csv():
    dtc = DecisionTreeClassifier(max_depth=4)
    # fit the model to the TRAIN dataset:
    dtc.fit(X_bow, y_train)
    dtc.score(X_bow,y_train)
    dtc_preds = dtc.predict(X_bow)
    pd.crosstab(dtc_preds,y_train)
    # as with any other sklearn transformation, 
    # transform only on our validate and/or test, 
    # only fit on train
    X_test_bow = tfidf.transform(X_test)
    dtc_preds_test = dtc.predict(X_test_bow)
    X_validate_bow = tfidf.transform(X_validate)
    dtc.score(X_validate_bow, y_validate)
    pred_df = pd.DataFrame({'is_R':dtc_preds_test})
    pred_df.to_csv('pred_test_set.csv',index=False)
    
    print(f'Accuracy-Train {round(dtc.score(X_bow,y_train),4)}')
    print(f'Accuracy-test {round(dtc.score(X_test_bow,y_test),4)}')
    print(classification_report(y_train,dtc_preds))
    print(classification_report(y_test,dtc.predict(X_test_bow)))