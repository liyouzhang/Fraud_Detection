from feature_engineering import feature_engineer, create_y
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer

# from bs4 import BeautifulSoup
# import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score


#1. read data
df = pd.read_json('data/data.json')
mindf = feature_engineer(df)
y = create_y(df)

#2. split train, test data
X_train, X_test, y_train, y_test = train_test_split(mindf, y ,test_size=0.2,random_state=17)

#3. basic model - logistic regression
lg = linear_model.LogisticRegression()
model = lg.fit(X_train,y_train)
y_hat_test = model.predict(X_test)

#4. measure of performance
# [TP,FP][FN,TN] = confusion_matrix(y_test,y_hat_test)
print ('accuracy:', accuracy_score(y_test,y_hat_test))
print ('precision:',precision_score(y_test,y_hat_test))
print ('recall:',recall_score(y_test,y_hat_test))

#5. pickle the model
with open('model.pkl', 'wb') as f:
    # Write the model to a file.
    pickle.dump(model, f)

# if __name__ == '__main__':
    # while True:
    #     raw_data = get_data()
    #     print("hello")
    #     df = pd.read_json('newdata.json')
    #     print('hello2')
    #     prediction = inference(df)
    #     print(prediction)
    #     time.sleep(0.5)