import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import gc
import riiideducation
env = riiideducation.make_env()

dir_path = '/kaggle/input/riiid-test-answer-prediction/'
file_train = 'train.csv'
file_questions = 'questions.csv'
file_lectures = 'lectures.csv'

nrows = 100 * 10000
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

lectures = pd.read_csv(
                       dir_path + file_lectures, 
                       usecols=['lecture_id','tag','part','type_of'], 
                       nrows=nrows,
                       dtype={
                           'lecture_id': 'int16',
                           'tag': 'int16',
                           'part': 'int8',
                           'type_of': 'str'
                       }
                    )

questions = pd.read_csv(
                        dir_path + file_questions, 
                        nrows=nrows,
                        usecols=['question_id','bundle_id','part','tags'], 
                        dtype={
                           'question_id': 'int16',
                           'bundle_id': 'int16',
                           'part': 'int8',
                           'tags': 'str'
                       }
                    )

# 数据处理
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)
lectures['type_of'] = lectures['type_of'].map({'concept':0, 'intention':1, 'solving question':2, 'starter':3}).fillna(-1).astype(np.int8)
questions['tags'] = questions['tags'].map(lambda x:len(str(x).split(' ')))

# 压缩内存
max_num = 1000
train = train.groupby(['user_id']).tail(max_num)


# 切分数据
train_lectures = train[train['content_type_id']==1]
train_questions = train[train['content_type_id']==0]
del train
gc.collect()

# 关联数据
train_lectures_info = pd.merge(
        left=train_lectures,
        right=lectures,
        how='left',
        left_on='content_id',
        right_on='lecture_id'
        )

train_questions_info = pd.merge(
        left=train_questions,
        right=questions,
        how='left',
        left_on='content_id',
        right_on='question_id'
        )

del train_lectures
del train_questions
gc.collect()


# 提取特征函数
# 文献课程类函数
def get_lecture_basic_features__user(train_lectures_info):
    gb_columns = ['user_id']
    gb_suffixes = 'lecture_'+'_'.join(gb_columns)
    
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],

        # type_of 展开
        'type_of': [lambda x: len(set(x))],
    }
    columns = [
           gb_suffixes+'_size_lecture_id', 
           gb_suffixes+'_unique_task_container_id',
           gb_suffixes+'_unique_tag',
           gb_suffixes+'_unique_part',
           gb_suffixes+'_unique_type_of'
          ]  
    train_lectures_info__user_f = train_lectures_info.\
                                groupby(gb_columns).\
                                agg(agg_func).\
                                reset_index()
    
    train_lectures_info__user_f.columns = gb_columns + columns
    return train_lectures_info__user_f

def get_lecture_basic_features__user_tag(train_lectures_info):
    gb_columns = ['user_id','tag']
    gb_suffixes = 'lecture_'+'_'.join(gb_columns)
    agg_func = {
        'lecture_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],
        'tag': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_size_lecture_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_unique_tag',
               gb_suffixes+'_unique_part'
              ]    
    train_lectures_info__user_tag_f = train_lectures_info.\
                                    groupby(gb_columns).\
                                    agg(agg_func).\
                                    reset_index()
    train_lectures_info__user_tag_f.columns = gb_columns + columns    
    return train_lectures_info__user_tag_f

# 问答类函数
def get_questions_basic_features__user(train_questions_info):
    gb_columns = ['user_id']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],

        'bundle_id': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
        'tags': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_question_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',

               gb_suffixes+'_unique_bundle_id',
               gb_suffixes+'_unique_part',
               gb_suffixes+'_unique_tags',
              ]
    train_questions_info__user_f = train_questions_info.\
                                    groupby(gb_columns).\
                                    agg(agg_func).\
                                    reset_index()
    train_questions_info__user_f.columns = gb_columns + columns    

    return train_questions_info__user_f

def get_questions_basic_features__user_part(train_questions_info):
    gb_columns = ['user_id','part']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'question_id': [np.size],
        'task_container_id': [lambda x: len(set(x))],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],

        'bundle_id': [lambda x: len(set(x))],

        # part 展开
        'part': [lambda x: len(set(x))],
        'tags': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_question_id', 
               gb_suffixes+'_unique_task_container_id',
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',

               gb_suffixes+'_unique_bundle_id',
               gb_suffixes+'_unique_part',
               gb_suffixes+'_unique_tags',
              ]    
    train_questions_info__user_part_f = train_questions_info.\
                                    groupby(gb_columns).\
                                    agg(agg_func).\
                                    reset_index()
    train_questions_info__user_part_f.columns = gb_columns + columns    

    return train_questions_info__user_part_f

def get_questions_basic_features__content(train_questions_info):
    gb_columns = ['content_id']
    gb_suffixes = 'question_'+'_'.join(gb_columns)
    agg_func = {
        'answered_correctly': [np.mean,np.sum,np.std],

        'user_id': [np.size],

        'prior_question_elapsed_time': [np.mean,np.max,np.min],

        'prior_question_had_explanation': [lambda x: len(set(x))],
    }
    columns = [
               gb_suffixes+'_answered_correctly_mean',
               gb_suffixes+'_answered_correctly_max',
               gb_suffixes+'_answered_correctly_min',

               gb_suffixes+'_size_user_id', 
               gb_suffixes+'_prior_question_elapsed_time_mean',
               gb_suffixes+'_prior_question_elapsed_time_max',
               gb_suffixes+'_prior_question_elapsed_time_min',

               gb_suffixes+'_unique_prior_question_had_explanation',
              ]    
    
    train_questions_info__user_content_f = train_questions_info.\
                                    groupby(gb_columns).\
                                    agg(agg_func).\
                                    reset_index()
    train_questions_info__user_content_f.columns = gb_columns + columns
    
    return train_questions_info__user_content_f


# 预测用户
test_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
# test_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
test_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
# test_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
test_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

# 验证数据
valid_data = pd.DataFrame()

for i in range(3):
    
    # 获取训练标签数据
    last_records = train_questions_info.drop_duplicates('user_id', keep='last')
    
    # 获取训练标签以前的数据
    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))
    
    train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
    train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)

    train_questions_info = train_questions_info[train_questions_info['row_id']<train_questions_info['filter_row']]
    train_lectures_info = train_lectures_info[train_lectures_info['row_id']<train_lectures_info['filter_row']]
    
    # 获取特征
    train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
    # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
    train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
    # train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
    train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

    last_records = last_records.merge(train_lectures_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_content_f,on=['content_id'],how='left')
    
    # 特征加入训练集
    valid_data = valid_data.append(last_records)
    print(len(valid_data))
    

# 训练数据
train_data = pd.DataFrame()

for i in range(10):
    
    # 获取训练标签数据
    last_records = train_questions_info.drop_duplicates('user_id', keep='last')
    
    # 获取训练标签以前的数据
    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))
    
    train_questions_info['filter_row'] = train_questions_info['user_id'].map(map__last_records__user_row)
    train_lectures_info['filter_row'] = train_lectures_info['user_id'].map(map__last_records__user_row)

    train_questions_info = train_questions_info[train_questions_info['row_id']<train_questions_info['filter_row']]
    train_lectures_info = train_lectures_info[train_lectures_info['row_id']<train_lectures_info['filter_row']]
    
    # 获取特征
    train_lectures_info__user_f = get_lecture_basic_features__user(train_lectures_info)
    # train_lectures_info__user_tag_f = get_lecture_basic_features__user_tag(train_lectures_info)
    train_questions_info__user_f = get_questions_basic_features__user(train_questions_info)
    # train_questions_info__user_part_f = get_questions_basic_features__user_part(train_questions_info)
    train_questions_info__user_content_f = get_questions_basic_features__content(train_questions_info)

    last_records = last_records.merge(train_lectures_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_f,on=['user_id'],how='left')
    last_records = last_records.merge(train_questions_info__user_content_f,on=['content_id'],how='left')
    
    # 特征加入训练集
    train_data = train_data.append(last_records)
    print(len(train_data))


# 修改
remove_columns = ['user_id','row_id','content_type_id','user_answer','answered_correctly','filter_row']
features_columns = [c for c in train_data.columns if c not in remove_columns]

X_test, y_test = valid_data[features_columns].values, valid_data['answered_correctly'].values

X_train, y_train = train_data[features_columns].values, train_data['answered_correctly'].values


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 9,
    'learning_rate': 0.3,
    'feature_fraction_seed': 2,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data': 20,
    'min_hessian': 1,
    'verbose': -1,
    'silent': 0
    }

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(
			params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=lgb_eval,
            early_stopping_rounds=20
            )

iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:
    
    test_questions = test_df[test_df['content_type_id']==0]
    test_questions_info = pd.merge(
            left=test_questions,
            right=questions,
            how='left',
            left_on='content_id',
            right_on='question_id'
            )
    
    test_questions_info['prior_question_had_explanation'] = test_questions_info['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

    test_questions_info = test_questions_info.merge(test_lectures_info__user_f,on=['user_id'],how='left')
    test_questions_info = test_questions_info.merge(test_questions_info__user_f,on=['user_id'],how='left')
    test_questions_info = test_questions_info.merge(test_questions_info__user_content_f,on=['content_id'],how='left')
        
    # 修改
    #remove_columns = ['user_id','row_id','content_type_id','user_answer','answered_correctly','filter_row']
    #features_columns = [c for c in train_data.columns if c not in remove_columns]


    X_test = test_questions_info[features_columns].values
    
    test_questions_info['answered_correctly'] =  gbm.predict(X_test)
    
    env.predict(test_questions_info.loc[test_questions_info['content_type_id'] == 0, ['row_id', 'answered_correctly']])


