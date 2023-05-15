import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("tab10")
from scipy import stats
from sklearn.model_selection import train_test_split
import os
from langdetect import detect
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
