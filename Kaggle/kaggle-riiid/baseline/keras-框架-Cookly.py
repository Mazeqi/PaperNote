import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import gc
import riiideducation
env = riiideducation.make_env()

dir_path = '/kaggle/input/riiid-test-answer-prediction/'
file_train = 'train.csv'
file_questions = 'questions.csv'

nrows =  100 * 10000
# nrows = None

train = pd.read_csv(
                    dir_path + file_train, 
                    nrows=nrows, 
                    usecols=['row_id', 'timestamp', 'user_id', 'content_id', 
                             'content_type_id', 'task_container_id', 'answered_correctly',
                            'prior_question_elapsed_time','prior_question_had_explanation'],
                    dtype={
                            'row_id': 'int64',
                            'timestamp': 'int64',
                            'user_id': 'int32',
                            'content_id': 'int16',
                            'content_type_id': 'int8',
                            'task_container_id': 'int8',
                            'answered_correctly': 'int8',
                            'prior_question_elapsed_time': 'float32',
                            'prior_question_had_explanation': 'str'
                        }
                   )

questions = pd.read_csv(
                        dir_path + file_questions, 
                        nrows=nrows,
                        usecols=['question_id','bundle_id','part'], 
                        dtype={
                           'question_id': 'int16',
                           'bundle_id': 'int16',
                           'part': 'int8',
                       }
                    )


train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

train = train[train['content_type_id']==0]

gc.collect()

# 压缩内存
max_num = 100
train = train.groupby(['user_id']).tail(max_num)

train = pd.merge(
        left=train,
        right=questions,
        how='left',
        left_on='content_id',
        right_on='question_id'
        )

train = train.fillna(0)

class cat_deal:
    def __init__(self):
        self.max_len = 0
        self.dict_map = {}
    
    def fit(self, cat_list):
        index = 1 
        for cat_i in cat_list:
            if cat_i not in self.dict_map:
                self.dict_map[cat_i] = index
                index += 1
        self.max_len = index + 1
        
    def transform(self, cat_list):
        cat_transform_list = []
        for cat_i in cat_list:
            if cat_i in self.dict_map:
                cat_transform_list.append(self.dict_map[cat_i])
            else:
                cat_transform_list.append(0)
        return cat_transform_list

class float_deal:
    def __init__(self):
        self.max = 0
        self.min = 0
        self.max_min = 0 
        
    def fit(self, float_list):
        for float_i in float_list:
            if float_i < self.min:
                self.min = float_i
            if float_i > self.max:
                self.max = float_i
        self.max_min = self.max - self.min
        
    def transform(self, float_list):
        float_transform_list = []
        for float_i in float_list:
            if float_i < self.min:
                float_transform_list.append(0)
            elif float_i > self.max:
                float_transform_list.append(1)
            else:
                float_transform_list.append(float_i/self.max_min)
        return float_transform_list

dict_cat_class = {}
for columns in ['user_id','content_id',\
                'task_container_id','prior_question_had_explanation',\
                'bundle_id','part']:
    dict_cat_class[columns] = cat_deal()
    dict_cat_class[columns].fit(train[columns])

    train[columns] = dict_cat_class[columns].transform(train[columns])
    print(columns)


dict_float_class = {}
for columns in ['timestamp','prior_question_elapsed_time']:
    dict_float_class[columns] = float_deal()
    dict_float_class[columns].fit(train[columns])
    
    train[columns] = dict_float_class[columns].transform(train[columns])
    print(columns)

def squeeze(embedding):
    embedding = tf.squeeze(embedding,axis=1)
    return embedding

def concat(embedding_list):
    embedding = tf.concat(embedding_list, axis=1)
    return embedding
    
def multiply(multi_x_y):
    multi_x = multi_x_y[0]
    multi_y = multi_x_y[1]
    multi_x_y = tf.multiply(multi_x, multi_y)
    return multi_x_y

# 模型
input_timestamp = tf.keras.Input(shape=(1,))
input_prior_question_elapsed_time = tf.keras.Input(shape=(1,))

# input int
input_user = tf.keras.Input(shape=(1,))
input_content = tf.keras.Input(shape=(1,))
input_task_container = tf.keras.Input(shape=(1,))
input_prior_question_had_explanation = tf.keras.Input(shape=(1,))
input_bundle = tf.keras.Input(shape=(1,))
input_part = tf.keras.Input(shape=(1,))

inputs = [input_timestamp,input_prior_question_elapsed_time,\
         input_user,input_content,\
         input_task_container,input_prior_question_had_explanation,\
         input_bundle,input_part]
# inputs = tf.keras.layers.Lambda(concat)(inputs)

# input session
# input_tags = Input(shape=(1))

# embedding float
embedding_timestamp = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(input_timestamp)
embedding_prior_question_elapsed_time = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(input_prior_question_elapsed_time)

# embedding int 
embedding_user = tf.keras.layers.Embedding(dict_cat_class['user_id'].max_len,
                                           64, input_length=1)(input_user)
embedding_user = tf.keras.layers.Lambda(squeeze)(embedding_user)

embedding_content = tf.keras.layers.Embedding(dict_cat_class['content_id'].max_len,
                                              64, input_length=1)(input_content)
embedding_content = tf.keras.layers.Lambda(squeeze)(embedding_content)

embedding_task_container = tf.keras.layers.Embedding(dict_cat_class['task_container_id'].max_len,
                                                     64, input_length=1)(input_task_container)
embedding_task_container = tf.keras.layers.Lambda(squeeze)(embedding_task_container)

embedding_prior_question_had_explanation = tf.keras.layers.Embedding(dict_cat_class['prior_question_had_explanation'].max_len, 
                                                                     64, input_length=1)(input_prior_question_had_explanation)
embedding_prior_question_had_explanation = tf.keras.layers.Lambda(squeeze)(embedding_prior_question_had_explanation)

embedding_bundle = tf.keras.layers.Embedding(dict_cat_class['bundle_id'].max_len,
                                             64, input_length=1)(input_bundle)
embedding_bundle = tf.keras.layers.Lambda(squeeze)(embedding_bundle)

embedding_part = tf.keras.layers.Embedding(dict_cat_class['part'].max_len,
                                           64, input_length=1)(input_part)
embedding_part = tf.keras.layers.Lambda(squeeze)(embedding_part)

embedding_all = [embedding_timestamp,embedding_prior_question_elapsed_time,\
                embedding_user, embedding_content, embedding_task_container,\
                embedding_prior_question_had_explanation, embedding_bundle, embedding_part]


nffm1, nffm2 = [], []
for i, embedding_i in enumerate(embedding_all):
    for j, embedding_j in enumerate(embedding_all):
        if i > j:
            nffm1.append(embedding_i), nffm2.append(embedding_j)
nffm1_layer = tf.keras.layers.Lambda(concat)(nffm1)
nffm2_layer = tf.keras.layers.Lambda(concat)(nffm2)     

nffm_all = tf.keras.layers.Lambda(multiply)([nffm1_layer,nffm2_layer])
    
logit = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(nffm_all)


model = tf.keras.models.Model(inputs=inputs, outputs=logit)

# 编译模型 设置参数
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_crossentropy'])

plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                            verbose=0,
                            mode='min',
                            factor=0.1,
                            patience=6)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                               verbose=0,
                               mode='min',
                               patience=10)

# 保存
checkpoint = tf.keras.callbacks.ModelCheckpoint(f'fold.h5',
                             monitor='val_loss',
                             verbose=0,
                             mode='min',
                             save_best_only=True)


# 训练 验证
valid = pd.DataFrame()
for i in range(6):
    
    # 获取训练标签数据
    last_records = train.drop_duplicates('user_id', keep='last')
    
    # 获取训练标签以前的数据
    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))
    train['filter_row'] = train['user_id'].map(map__last_records__user_row)
    train = train[train['row_id']<train['filter_row']]

    # 特征加入训练集
    valid = valid.append(last_records)
    print(len(valid))


features_columns = ['timestamp','prior_question_elapsed_time',\
                    'user_id','content_id',\
                    'task_container_id','prior_question_had_explanation',\
                    'bundle_id','part']

X_valid, y_valid = [valid[columns].values for columns in features_columns], valid['answered_correctly'].values
del valid

X_train, y_train = [train[columns].values for columns in features_columns], train['answered_correctly'].values
del train


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


model.fit(X_train, y_train,
          epochs=1,
          batch_size=512 * 500 * 2,
          verbose=1,
          shuffle=True,
          validation_data=(X_valid, y_valid),
          callbacks=[plateau, early_stopping, checkpoint])

y_valid_proba = model.predict(X_valid, verbose=0, batch_size=512)
auc = roc_auc_score(y_valid, y_valid_proba)
print(auc)


iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

	test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

	test_df = pd.merge(
        left=test_df,
        right=questions,
        how='left',
        left_on='content_id',
        right_on='question_id'
        )

	test_df = test_df.fillna(0)


	for columns in ['user_id','content_id',\
	                'task_container_id','prior_question_had_explanation',\
	                'bundle_id','part']:

	    test_df[columns] = dict_cat_class[columns].transform(test_df[columns])
	    print(columns)


	for columns in ['timestamp','prior_question_elapsed_time']:
	  
	    test_df[columns] = dict_float_class[columns].transform(test_df[columns])
	    print(columns)

	X_test = [test_df[columns].values for columns in features_columns]

	test_df['answered_correctly'] =  model.predict(X_test, verbose=0, batch_size=512)
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])






