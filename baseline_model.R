library(tidyverse)
library(dbplyr)
library(tidytext)
library(foreach)
library(magrittr)
library(lightgbm)
library(Matrix)
library(tictoc)
library(doParallel)
registerDoParallel(4)

DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'

TRAIN_PARAMS =  list(
    'task'= 'train',
    'boosting_type'= 'gbdt',
    'objective'= 'regression',
    'metric'= 'rmse',
    'max_depth'= 5,
    'num_leaves'= 31,
    'feature_fraction'= 0.80,
    'bagging_fraction'= 0.90,
    'bagging_freq'= 5,
    'lambda_l2'= 5,
    'verbose'= 0
)

N_ROUNDS = 9041

CATEGORICAL_FEATURES <- c(
    'region',
    'city',
    'parent_category_name',
    'category_name',
    'param_1',
    'param_2',
    'param_3',
    'activation_date',
    'user_type',
    'image_top_1',
    'no_img',
    'no_dsc'
)

VAL_DATES = c(17275,17276)

# File I/O
tic('Loading Data')
INPUT_DB <-
    file.path(DATA_FOLDER, 'derived', INPUT_DB) %>%
    src_sqlite(create=TRUE)

train <- 
    INPUT_DB %>% 
    tbl('train') %>%
    dplyr::select(-item_id, -image, -title, -description, -user_id) %>%
    collect()

test <-
    INPUT_DB %>% 
    tbl('test') %>%
    dplyr::select(-item_id, -image, -title, -description, -user_id) %>%
    collect()

label <-
    INPUT_DB %>% 
    tbl('deal_probability') %>%
    select(-item_id) %>%
    collect()
toc()

# Preparing rules on test to (??) ignore features that irrelevant to test time
tic('prepare rules')
test_rules <- lgb.prepare_rules(test)
train_data <- lgb.prepare_rules(data = train, rules = test_rules$rules)$data
toc()

# Split into Train/Val and convert into Dataset format
tic('data reorganization')
val_label <- label[train_data$activation_date %in% VAL_DATES,]$deal_probability
train_label <- label[!(train_data$activation_date %in% VAL_DATES),]$deal_probability

val_data <- 
    train_data %>% 
    filter(activation_date %in% VAL_DATES) %>%
    as.matrix

train_data <- 
    train_data %>% 
    filter(!(activation_date %in% VAL_DATES)) %>%
    as.matrix

train_data <- lgb.Dataset(data = train_data,
                          label = train_label)

val_data <- lgb.Dataset(data = val_data,
                        label = val_label)
toc()

model <- lgb.train(
    params=TRAIN_PARAMS, 
    train_data,
    nrounds=N_ROUNDS,
    min_data=1,
    early_stopping_rounds=200,
    categorical_feature = which(colnames(train_data) %in% CATEGORICAL_FEATURES),
    valids = list(train = train_data, valid = val_data)
)