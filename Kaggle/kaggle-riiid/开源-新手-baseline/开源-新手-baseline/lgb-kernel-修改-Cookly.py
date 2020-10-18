import riiideducation
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
env = riiideducation.make_env()

nrows = 100 * 10000
# nrows = None

train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                   nrows=nrows, 
                   usecols=[1, 2, 3, 4, 7, 8, 9],
                   dtype={'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )

train = train[train.content_type_id == False]

train = train.sort_values(['timestamp'], ascending=True).reset_index(drop = True)

results_c_final = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','sum','count','std'])
results_c_final.columns = ["content_y_mean","content_y_sum","content_y_count","content_y_std"]

results_u_final = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum', 'count','std'])
results_u_final.columns = ['user_y_mean', 'user_y_sum', 'user_y_count','user_y_std']

results_ct_final = train[['content_id','prior_question_elapsed_time']].groupby(['content_id']).agg(['mean', 'sum', 'count','std'])
results_ct_final.columns = ['content_t_mean', 'content_t_sum', 'content_t_count','content_t_std']

results_ut_final = train[['user_id','prior_question_elapsed_time']].groupby(['user_id']).agg(['mean', 'sum', 'count','std'])
results_ut_final.columns = ['user_t_mean', 'user_t_sum', 'user_t_count','user_t_std']

train.drop(['timestamp', 'content_type_id'], axis=1, inplace=True)


validation = pd.DataFrame()

for i in range(6):
    last_records = train.drop_duplicates('user_id', keep = 'last')
    train = train[~train.index.isin(last_records.index)]
    validation = validation.append(last_records)
    print('validation : ', i)


X = pd.DataFrame()

for i in range(30):
    last_records = train.drop_duplicates('user_id', keep = 'last')
    train = train[~train.index.isin(last_records.index)]
    X = X.append(last_records)
    print('X : ', i)

results_c_tv = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','sum','count','std'])
results_c_tv.columns = ["content_y_mean","content_y_sum","content_y_count","content_y_std"]

results_u_tv = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum', 'count','std'])
results_u_tv.columns = ['user_y_mean', 'user_y_sum', 'user_y_count','user_y_std']

results_ct_tv = train[['content_id','prior_question_elapsed_time']].groupby(['content_id']).agg(['mean', 'sum', 'count','std'])
results_ct_tv.columns = ['content_t_mean', 'content_t_sum', 'content_t_count','content_t_std']

results_ut_tv = train[['user_id','prior_question_elapsed_time']].groupby(['user_id']).agg(['mean', 'sum', 'count','std'])
results_ut_tv.columns = ['user_t_mean', 'user_t_sum', 'user_t_count','user_t_std']

del(train)

X = pd.merge(X, results_u_tv, on=['user_id'], how="left")
print(X.columns)
X = pd.merge(X, results_c_tv, on=['content_id'], how="left")
print(X.columns)
X = pd.merge(X, results_ut_tv, on=['user_id'], how="left")
print(X.columns)
X = pd.merge(X, results_ct_tv, on=['content_id'], how="left")
print(X.columns)

validation = pd.merge(validation, results_u_tv, on=['user_id'], how="left")
validation = pd.merge(validation, results_c_tv, on=['content_id'], how="left")
validation = pd.merge(validation, results_ut_tv, on=['user_id'], how="left")
validation = pd.merge(validation, results_ct_tv, on=['content_id'], how="left")

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

X.prior_question_had_explanation.fillna(False, inplace = True)
validation.prior_question_had_explanation.fillna(False, inplace = True)

validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])
X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])

y = X['answered_correctly']
X = X.drop(['answered_correctly'], axis=1)

y_val = validation['answered_correctly']
X_val = validation.drop(['answered_correctly'], axis=1)

# X.columns

columns_features = [c for c in X.columns if c not in ['user_id','content_id','prior_question_had_explanation']]

X = X[columns_features]
X_val = X_val[columns_features]

X.fillna(-1,  inplace=True)
X_val.fillna(-1, inplace=True)

import lightgbm as lgb

params = {
    'objective': 'binary',
    'max_bin': 600,
    'learning_rate': 0.02,
    'num_leaves': 80
}

lgb_train = lgb.Dataset(X, y)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)


model = lgb.train(
    params, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=25,
    num_boost_round=10000,
    early_stopping_rounds=10
)


y_pred = model.predict(X_val)
y_true = np.array(y_val)
roc_auc_score(y_true, y_pred)


iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:
	test_df = pd.merge(test_df, results_u_final, on=['user_id'],  how="left")
	test_df = pd.merge(test_df, results_c_final, on=['content_id'],  how="left")
	test_df = pd.merge(test_df, results_ut_final, on=['user_id'],  how="left")
	test_df = pd.merge(test_df, results_ct_final, on=['content_id'],  how="left")

    test_df['prior_question_had_explanation'].fillna(False, inplace=True)
    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])

    X_test = test_df[columns_features]
    X_test.fillna(-1, inplace=True)

    test_df['answered_correctly'] =  model.predict(X_test)
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])


