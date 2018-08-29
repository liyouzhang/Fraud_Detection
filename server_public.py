import os
import pickle
from flask import Flask,render_template, request,jsonify,Response
import pandas as pd
import time
from src.feature_engineering import feature_engineer
import json
from pandas.io.json import json_normalize
from flask_pymongo import PyMongo

app = Flask(__name__)


###########################

@app.route('/dash_table', methods = ['GET'])
def dash_table():

    df = pd.read_json('newdata.json')
    df = json_normalize(list(df.data))
    mindf = feature_engineer(df)
    y_hat = model.predict_proba(mindf)
    with open ('prediction.csv','w') as f:
        f.write(str(y_hat[0][1]))
    
    prob = pd.read_csv('prediction.csv',header=None,names=['probability'])
    p = prob.probability
    df['probability'] = p
    
    if p[0] > 0.8:
        df['label'] = 'high risk'
    elif p[0] <= 0.8 and p[0] >0.6:
        df['label'] = 'middle risk'
    elif p[0] <= 0.6 and p[0] >=0.5:
        df['label'] = 'low risk'
    # if 'label' in df.keys():
    else:
        df['label'] = 'Not Fraud'
    df.to_json('df_predictions.json')

    from pymongo import MongoClient
    mongo_client = MongoClient("mongodb://lambo5:lambo5@54.237.222.133/test")
    db = mongo_client.test
    # coll = db.fraud
    # coll.insert_one(data)
    os.system("mongoimport --db test --collection fraud --file df_predictions.json --username lambo5 --password lambo5")


    country = df.country.values
    null = df.country.isnull()
    country[null] = 0

    name = df.name.values
    null = df.name.isnull()
    name[null] = 0

    currency = df.currency.values
    null = df.currency.isnull()
    currency[null] = 0


    currency = df['org_name'].values
    null = df['org_name'].isnull()
    currency[null] = 0

    sold_amount = df['sold_amount'].values
    null = df['sold_amount'].isnull()
    sold_amount[null] = 0

    data = {'probability':float("{:10.3f}".format(df['probability'][0]*100).strip()),
    'country':df.country[0],
    'name':df.name[0],
    'currency':df.currency[0],
    'org_name':df['org_name'][0],
    'sold_amount':df["sold_amount"][0],
    'max_sales':float("{:10.0f}".format(df['max_sales'][0]).strip()),
    'label':df["label"][0]}
    return jsonify(data)


###########################


#Need to load in the model here
with open('model_Ada.pkl', 'rb') as f:
    model = pickle.load(f)


###########################

import json
from bson import ObjectId

# Route for getting new data
@app.route('/new_data',methods = ['GET'])
def get_data():
    import requests
    api_key = 'PUT_API_KEY_HERE'
    url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
    sequence_number = 0
    response = requests.post(url, json={'api_key': api_key,
                                        'sequence_number': sequence_number})
    raw_data = response.json()
    with open ('newdata.json','w') as f:
        f.write(json.dumps(raw_data))

    return jsonify(raw_data)



if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)
