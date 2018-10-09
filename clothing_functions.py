#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:47:34 2018
Functions used in Women’s Clothing E-Commerce Projects
@author: hanzhang
"""
# functions for the women E-Commerce review sentiment analysis 
import seaborn as sns
import nltk 
import string

from wordcloud import WordCloud
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_cloud(words_dist,size):
    """
    Plot the wordcloud figure
    """
    mpl.rcParams['figure.figsize'] = (10.0, 10.0)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.subplot.bottom'] = .1
    
    wordcloud = WordCloud(width=1600, height=800,
                          background_color='black').generate_from_frequencies(words_dist)
    
    fig = plt.figure(figsize=size, facecolor='k')
    plt.imshow(wordcloud)
    plt.axis('off')
    #plt.title(title, fontsize=50, color='y')
    plt.tight_layout()
   # plt.savefig('{}.png'.format(title), format='png', dpi=300)
    plt.show()

def plot_dist(data,column,ax=None):
    if ax is None: 
        sns.distplot(data[column], color="b", label="Skewness : %.2f  Kurtosis : %.2f"\
                %(data[column].skew(),data[column].kurt()),bins=25, kde=False)    
    else:
        sns.distplot(data[column], color="b", label="Skewness : %.2f  Kurtosis : %.2f"\
                %(data[column].skew(),data[column].kurt()),bins=25, ax=ax, kde=False)  
        
    ax.legend(loc="best")
    return None

# a number "a" from the vector "x" is an outlier if 
# a > median(x)+1.5*iqr(x) or a < median-1.5*iqr(x)
# iqr: interquantile range = third interquantile - first interquantile
# replace outlier in Age by 0.05 or 0.95 quantile
def outliers(x): 
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    out={'lower':(x < (Q1 - 1.5 * IQR)), 'higher':(x > (Q3 + 1.5 * IQR))}
    return out

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

# remove it/have/has etc 
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')
#clean_review = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#from nltk.stem import SnowballStemmer,WordNetLemmatizer
#stemmer=SnowballStemmer('english')
#lemma=WordNetLemmatizer() # worked the same way as tonized 

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    nopunc = ([char for char in mess.lower() \
               if char not in string.punctuation and not char.isdigit()])
    nopunc = ''.join(nopunc)
    new_word=[((word)) for word in nopunc.split() if word not in stopwords_en]
        #new_word=[stemmer.stem(word) for word in new_word.split()]
    return new_word 

class clothing_text:
    def __init__(self,s):
        self.s=s
       # self.s = s

    def word_freq(self):
        txt = self.s.str.lower().str.replace(r'\W+', ' ').str.cat(sep=' ')  
        words = text_process(txt)
        words_dist = nltk.FreqDist(w for w in words) 
        return words_dist
   


