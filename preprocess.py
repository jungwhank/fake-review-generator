#!/usr/bin/env python3
# coding=utf-8
"""
Preprocess of your Yelp Datatset
"""

import json
import pandas as pd
import re
import os

# Input and Output path
PATH = os.getcwd() + '/yelp_dataset/'
OUT_FPATH = os.getcwd() + '/yelp_dataset/preprocess/'

if not os.path.exists(OUT_FPATH):
    os.mkdir(OUT_FPATH)

# Review file is large. so choose max number of output files.
MAX_REVIEW_NUM = 500000


def preprocess_business():
    print('preprocessing business file.')
    business_path = PATH + 'business.json'
    df = pd.read_json(business_path, lines=True)

    # only keep necessary columns
    keeps = ['name', 'categories', 'business_id']
    df = df[keeps]

    # rename a column
    df.rename(columns={'name' : 'business_name'}, inplace=True)

    # keep only restaurant
    df = df.dropna(subset=['categories'])
    df = df[df['categories'].str.contains('Restaurants')]
    
    print('business len:', len(df))
    print('business preprocess done!')
    return df


def preprocess_review():
    print('preprocessing review file. It can take more than a few minutes.')
    review_path = PATH + 'review.json'
    df = pd.read_json(review_path, lines=True)

    # only keep necessary columns
    keeps = ['business_id', 'stars', 'text']
    df = df[keeps]
    
    # clean text
    def clean_text(text):
        text = re.sub('[-=+#/\?:^@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\…》]', '', text)
        text = re.sub('\n', '', text)
        return text

    df.text = df.text.apply(clean_text)

    print('review len:', len(df))
    print('review preprocess done!')
    return df


def preprocess(business_df, review_df):
    print('preprocessing')

    # merge review and restaurant
    df = business_df.merge(review_df, on='business_id')
    
    positive = df[df['stars'] > 3].text
    neutral = df[df['stars'] == 3].text
    negative = df[df['stars'] < 3].text

    # write to txt
    positive[:MAX_REVIEW_NUM].to_csv(OUT_FPATH + 'positive.txt', encoding='utf-8', header=None, index=None, sep=' ')
    neutral[:MAX_REVIEW_NUM].to_csv(OUT_FPATH + 'neutral.txt', encoding='utf-8', header=None, index=None, sep=' ')
    negative[:MAX_REVIEW_NUM].to_csv(OUT_FPATH + 'negative.txt', encoding='utf-8', header=None, index=None, sep=' ')

    print('pos text file:', len(positive[:MAX_REVIEW_NUM]))
    print('neu text file:', len(neutral[:MAX_REVIEW_NUM]))
    print('neg text file:', len(negative[:MAX_REVIEW_NUM]))

    print('preprocess done!')


def main():
    business = preprocess_business()
    review = preprocess_review()
    preprocess(business, review)


if __name__ == "__main__":
    main()