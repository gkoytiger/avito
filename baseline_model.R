library(tidyverse)
library(dbplyr)
library(tidytext)
library(foreach)
library(magrittr)
library(lightgbm)
library(Matrix)

DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'

INPUT_DB <-
    file.path(DATA_FOLDER, 'derived', INPUT_DB) %>%
    src_sqlite(create=TRUE)

train <- 
    INPUT_DB %>% 
    tbl('train') %>%
    select(-item_id, -image_id)
    collect()

label <-
    INPUT_DB %>% 
    tbl('deal_probability') %>%
    select(-item_id) %>%
    collect()

n_rounds = 9041
params =  list(
    'task'= 'train',
    'boosting_type'= 'gbdt',
    'objective'= 'regression',
    'metric'= 'rmse',
    'max_depth'= 15,
    'num_leaves'= 31,
    'feature_fraction'= 0.80,
    'bagging_fraction'= 0.90,
    'bagging_freq'= 5,
    'lambda_l2'= 5,
    'verbose'= 0
) 

prepared <- lgb.prepare_rules(train)$data
train_data

#train_data <- prepared %>% filter(activation_date %in% test_dates)
#test_data <- prepared %>% filter(!(activation_date %in% test_dates))

model <- 
    lgb.cv(params, 
           dtrain, 
           1, 
           nfold=5, 
           min_data=1,
           early_stopping_rounds=200,
           categorical_feature = c(2, 3, 4, 5, 7, 8, 9, 11, 16)))