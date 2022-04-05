import pandas as pd
from datetime import datetime
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings("ignore")


session = pd.read_csv("data/sessions.csv.zip")
train_user=pd.read_csv("data/train_users_2.csv.zip")
test_user=pd.read_csv("data/test_users.csv.zip")

#cleaning the train data variables
train_user['gender'] = [ s.replace('-', "") for s in train_user.gender]
train_user['first_browser'] = [ s.replace('-', "") for s in train_user.first_browser]
train_user['first_browser'] = [ s.replace(' ', "_") for s in train_user.first_browser]
train_user['first_browser'] = [ s.replace('.', "_") for s in train_user.first_browser]
train_user['first_device_type'] = [ s.replace(' ', "_") for s in train_user.first_device_type]
train_user['first_device_type'] = [ s.replace('/', "_") for s in train_user.first_device_type]
train_user['first_device_type'] = [ s.replace('(', "") for s in train_user.first_device_type]
train_user['first_device_type'] = [ s.replace(')', "") for s in train_user.first_device_type]
train_user['affiliate_channel'] = [ s.replace('-', "_") for s in train_user.affiliate_channel]
train_user['affiliate_provider'] = [ s.replace('-', "_") for s in train_user.affiliate_provider]

# Splitting date time data for date account created and date_first_booking
train_user['account_created_day'] = pd.DatetimeIndex(train_user['date_account_created']).day
train_user['account_created_month'] = pd.DatetimeIndex(train_user['date_account_created']).month
train_user['first_booking_day'] = pd.DatetimeIndex(train_user['date_first_booking']).day
train_user['first_booking_month'] = pd.DatetimeIndex(train_user['date_first_booking']).month

#Filling the NaN values by mean 
train_user.loc[train_user['age'] > 95, 'age'] = np.nan
train_user.loc[train_user['age'] < 16, 'age'] = np.nan
train_user.loc[train_user['age'].isnull(), 'age' ] = train_user['age'].median()

train_user = train_user.drop(['date_account_created', 'timestamp_first_active','date_first_booking','first_affiliate_tracked'], axis=1)
train_user.rename(columns={"id": "user_id"},inplace='True')

#--------------------------------------------------------------------------------------#

#cleaning the test data variables
test_user['gender'] = [ s.replace('-', "") for s in test_user.gender]
test_user['first_browser'] = [ s.replace('-', "") for s in test_user.first_browser]
test_user['first_browser'] = [ s.replace(' ', "_") for s in test_user.first_browser]
test_user['first_browser'] = [ s.replace('.', "_") for s in test_user.first_browser]
test_user['first_device_type'] = [ s.replace(' ', "_") for s in test_user.first_device_type]
test_user['first_device_type'] = [ s.replace('/', "_") for s in test_user.first_device_type]
test_user['first_device_type'] = [ s.replace('(', "") for s in test_user.first_device_type]
test_user['first_device_type'] = [ s.replace(')', "") for s in test_user.first_device_type]
test_user['affiliate_channel'] = [ s.replace('-', "_") for s in test_user.affiliate_channel]
test_user['affiliate_provider'] = [ s.replace('-', "_") for s in test_user.affiliate_provider]

# Splitting date time data for date account created and date_first_booking
test_user['account_created_day'] = pd.DatetimeIndex(test_user['date_account_created']).day
test_user['account_created_month'] = pd.DatetimeIndex(test_user['date_account_created']).month
test_user['first_booking_day'] = pd.DatetimeIndex(test_user['date_first_booking']).day
test_user['first_booking_month'] = pd.DatetimeIndex(test_user['date_first_booking']).month

#Filling the NaN values by mean 
test_user.loc[test_user['age'] > 95, 'age'] = np.nan
test_user.loc[test_user['age'] < 16, 'age'] = np.nan
test_user.loc[test_user['age'].isnull(), 'age' ] = test_user['age'].median()

test_user = test_user.drop(['date_account_created', 'timestamp_first_active','date_first_booking','first_affiliate_tracked'], axis=1)
test_user.rename(columns={"id": "user_id"},inplace='True')

######################################################################################

#cleaning the session data variables
#Handing the NaN values repaceing with mode of column
session['action'].fillna(session['action'].mode()[0], inplace=True)
session['action_type'].fillna(session['action_type'].mode()[0], inplace=True)
session['action_detail'].fillna(session['action_detail'].mode()[0], inplace=True)
session['device_type'].fillna(session['device_type'].mode()[0], inplace=True)
session['secs_elapsed'].fillna(0, inplace=True)

#replacing space with _ in device_type
session.device_type = session.device_type.str.replace(' ',"_")

#Gener#ating #new_session with one row for one user_id
user_id = session.groupby('user_id',sort = False)[['user_id']].apply(lambda x: list(np.unique(x)))
user_id.update(user_id.str[0])
action = session.groupby('user_id', sort = False)[['action']].sum()
action_type = session.groupby('user_id', sort = False)[['action_type']].sum()
action_detail = session.groupby('user_id', sort = False)[['action_detail']].sum()

#Taking mean value of particular user
sec_e= session.groupby('user_id', sort = False)[['secs_elapsed']].mean()

#Create new data frame as new_session after using groupby on session data
new_session=pd.DataFrame(action,columns=['action'])
new_session['action_type']=action_type['action_type'].values
new_session['action_detail']=action_detail['action_detail'].values
new_session['secs_elapsed']=sec_e['secs_elapsed'].values

#performing inner join on train and session data
train = pd.merge(train_user, new_session, left_on='user_id', right_on='user_id')
test = pd.merge(test_user, new_session, left_on='user_id', right_on='user_id')

#creating csv files
#train.to_csv('final_train.csv',  index=False)
#test.to_csv('final_test.csv',  index=False)

y = train['country_destination'].values
X= train.drop(['country_destination'], axis=1)

#####################################################################################

#Encoding categorical variables
vectorizer = CountVectorizer()
vectorizer.fit(X['gender'].values)
X_gender= vectorizer.transform(X['gender'].values)
test_gender= vectorizer.transform(test['gender'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['signup_method'].values)
X_signup_method = vectorizer.transform(X['signup_method'].values)
test_signup_method = vectorizer.transform(test['signup_method'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['language'].values)
X_language = vectorizer.transform(X['language'].values)
test_language = vectorizer.transform(test['language'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['affiliate_channel'].values)
X_affiliate_channel = vectorizer.transform(X['affiliate_channel'].values)
test_affiliate_channel = vectorizer.transform(test['affiliate_channel'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['affiliate_provider'].values)
X_affiliate_provider = vectorizer.transform(X['affiliate_provider'].values)
test_affiliate_provider = vectorizer.transform(test['affiliate_provider'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['signup_app'].values)
X_signup_app = vectorizer.transform(X['signup_app'].values)
test_signup_app = vectorizer.transform(test['signup_app'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['first_device_type'].values)
X_first_device_type = vectorizer.transform(X['first_device_type'].values)
test_first_device_type = vectorizer.transform(test['first_device_type'].values)

vectorizer = CountVectorizer()
vectorizer.fit(X['first_browser'].values)
X_first_browser = vectorizer.transform(X['first_browser'].values)
test_first_browser = vectorizer.transform(test['first_browser'].values)

#encoding session data
vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,4),max_features=1000)
vectorizer.fit(X['action'].values) 
X_action = vectorizer.transform(X['action'])
test_action = vectorizer.transform(test['action'])

vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,4),max_features=1000)
vectorizer.fit(X['action_type'].values) 
X_action_type = vectorizer.transform(X['action_type'])
test_action_type = vectorizer.transform(test['action_type'])

vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,4),max_features=1000)
vectorizer.fit(X['action_detail'].values) 
X_action_detail = vectorizer.transform(X['action_detail'])
test_action_detail = vectorizer.transform(test['action_detail'])


#Standardizing numerical features
scaler = preprocessing.StandardScaler().fit(X['secs_elapsed'].values.reshape(-1,1))
X_secs = scaler.transform(X['secs_elapsed'].values.reshape(-1,1))
test_secs = scaler.transform(test['secs_elapsed'].values.reshape(-1,1))

X_age=X['age'].values.reshape(-1,1)
X_flow=X['signup_flow'].values.reshape(-1,1)
X_create_day=X['account_created_day'].values.reshape(-1,1)
X_create_month=X['account_created_month'].values.reshape(-1,1)
X_book_day=X['first_booking_day'].values.reshape(-1,1)
X_book_month=X['first_booking_month'].values.reshape(-1,1)

test_age=test['age'].values.reshape(-1,1)
test_flow=test['signup_flow'].values.reshape(-1,1)
test_create_day=test['account_created_day'].values.reshape(-1,1)
test_create_month=test['account_created_month'].values.reshape(-1,1)
test_book_day=test['first_booking_day'].values.reshape(-1,1)
test_book_month=test['first_booking_month'].values.reshape(-1,1)

label_encoder = preprocessing.LabelEncoder() 
y_tr= label_encoder.fit_transform(y)
y_tr=y_tr.reshape(-1,1)

##################################################333

X_tr = hstack((X_gender, X_age, X_signup_method, X_flow, X_language,X_affiliate_channel,X_affiliate_provider,
              X_signup_app,X_first_device_type,X_first_browser,X_create_day,X_create_month,X_book_day,
              X_book_month,X_action,X_action_type,X_action_detail,X_secs)).tocsr()

final_test = hstack((test_gender, test_age, test_signup_method, test_flow, test_language,test_affiliate_channel,test_affiliate_provider,
              test_signup_app,test_first_device_type,test_first_browser,test_create_day,test_create_month,test_book_day,
              test_book_month,test_action,test_action_type,test_action_detail,test_secs)).tocsr()

###############################################3

def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


#def ndcg_score(ground_truth, predictions, k=5):
def ndcg_score(te_labels, predict, k):
    
    lb = LabelBinarizer()
    lb.fit(range(len(predict) + 1))
    T = lb.transform(te_labels)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predict):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        if best == 0:
            best = 0.000000001
        score = float(actual) / float(best)
        scores.append(score)
    return np.mean(scores)


# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, random_state=42)


##################################################3
n_estimator= [5,10,50,100] 
max_depth= [1,2,3,4]

for depth in max_depth:
    for n in n_estimator:       
        xgb = XGBClassifier(max_depth=depth, n_estimators=n,eval_metric='mlogloss')
        xgb.fit(X_train,y_train)
        prob_xgb = xgb.predict_proba(X_test)
        score_xgb = ndcg_score(y_test, prob_xgb, k=5)
        print ("Depth of tree : {}".format(depth))
        print ("Number of Estimators : {}".format(n))
        print ("NDCG Score : {}".format(score_xgb))
        print("*****************")


xgb = XGBClassifier(max_depth=1, n_estimators=100,eval_metric='mlogloss')
xgb.fit(X_train,y_train)
prob_xgb = xgb.predict_proba(X_test)
id_test = test.user_id
ids = []
countries = []

for i in range(len(prob_xgb)):
    idx = id_test[i]
    ids += [idx] * 5
    countries += label_encoder.inverse_transform(np.argsort(prob_xgb[i])[::-1])[:5].tolist()

output = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])
print(output.head())

output.to_csv('output.csv',  index=False)