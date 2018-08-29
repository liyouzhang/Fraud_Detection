import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from bs4 import BeautifulSoup
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score


def feature_engineer(df):
    #convert time columns into datetime format
    time_cols = ["user_created",'event_created', 'event_end', 'event_published', 'event_start',"user_created"]
    #"approx_payout_date" - not in live data
    for c in time_cols:
        df[c] = pd.to_datetime(df[c],unit='s')

    #create new feature - cost,quantity_sold,quantity_total
    def extract(x):
        cost = []
        for i in x:
            if 'quantity_sold' in i.keys():
                cost.append((i['cost'],i['quantity_sold'],i['quantity_total']))
            else:
                cost.append((i['cost'],0,i['quantity_total']))
        return cost
    df['cost_sold_total'] = df['ticket_types'].map(extract)

    #create new feature - total sold amount, max sales
    def total_amount(x):
        total_amount=0
        for i in x:
            amount,sold,_ = i
            total_amount += amount * sold
        return total_amount
    df['sold_amount'] = df['cost_sold_total'].map(total_amount)

    def max_amount(x):
        total_amount=0
        for i in x:
            amount,_,tosell = i
            total_amount += amount * tosell
        return total_amount
    df['max_sales'] = df['cost_sold_total'].map(max_amount)

    #create new feature - if currency and country is consistent
    euro_countries = ['AT', 'BE', 'CY', 'EE', 'FI', 'FR', 'DE', 'GR', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PT', 'SK', 'SI', 'ES']

    def currency_validity(country, currency):
        if country in euro_countries and currency == 'EUR':
            return 0
        elif country == currency[:2]:
            return 0
        else:
            return 1
    
    df['currency_not_c'] = df.country.combine(df.currency,func=currency_validity)

    #new feature - if event description is missing and if country is missing
    df['missing_desc'] = (df['description']== '')*1
    df['missing_country'] = (df['country'] == '')*1
    
    #new feature - if same day publish
    df['event_published_date'] = df['event_published'].apply(lambda x: x.date())
    df['user_created_date'] = df['user_created'].apply(lambda x: x.date())
    df['same_day_publish'] = (df['user_created_date'] == df['event_published_date'])*1

    # create timedelta payout - publish, payout - start

    #df['approx_payout_date'] = df['approx_payout_date'].apply(lambda x: x.date())
    #df['payout_publish'] = (df['approx_payout_date'] - df['event_published_date'])/np.timedelta64(1, 'D')
    #df['event_start_date'] = df['event_start'].apply(lambda x: x.date())
    #df['payout_start'] = (df['approx_payout_date'] - df['event_start_date'])/np.timedelta64(1, 'D')


    #logrify user_age
    df['user_age_lg'] = df['user_age'].apply(lambda x: np.log(x+1))
    df['org_facebook_lg'] = df['org_facebook'].apply(lambda x: np.log(x+1))


    #extract payout information
    def extract_payout(x):
        cost = []
        for i in x:
            if 'amount' in i.keys():
                cost.append((i['amount'],i['country']))#,i['created'],i['event'],i['name'],i['uid'],i['zip_code']
            else:
                cost.append(('N/A', i['country']))
        return cost
    df['payout_info'] = df['previous_payouts'].map(extract_payout)
    def payout_amount(x):
        payouts=0
        for i in x:
            amount,_ = i
            payouts += amount
        return payouts
    df['previous_payout_total'] = df['payout_info'].apply(payout_amount)

    #email_features
    df['email_org'] = df['email_domain'].apply(lambda x: "org" in x)*1
    df['email_common'] = (df['email_domain'].apply(lambda x: "gmail" in x or "hotmail" in x or "yahoo" in x or "live.com" in x ))*1


    min_features = ['same_day_publish',
                    'user_age_lg',
                    'missing_desc',
                    'email_common',
                    'org_facebook_lg',
                    'user_age',
                    'user_type',
                    'org_twitter',
                    'delivery_method',
                    'org_facebook',
                    'sale_duration',
                    'has_logo',
                    'channels',
                    'name_length',
                    'email_org',
                    'body_length',
                    'fb_published',
                    'has_analytics',
                    'has_header',
                    'show_map',
                    'venue_longitude',
                    'previous_payout_total',
                    'sold_amount',
                    'max_sales',
                    'object_id',
                    'missing_country',
                    'venue_latitude']
    #'sale_duration2' - not in live data

    mindf = df[min_features]
    mindf = mindf.fillna(0)
    return mindf


def create_y(df):
    df['fraud?'] = df["acct_type"].apply(lambda x: "fraud" in x)
    y = df['fraud?']*1
    return y
