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
OUTPUT_DB = 'submissions.sqlite'
MODEL_ID = 'baseline'

TRAIN_PARAMS =  list(
    'task'= 'train',
    'boosting_type'= 'gbdt',
    'objective'= 'regression',
    'metric'= 'rmse',
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
    tbl('clean_train') %>%
    select(-item_id) %>%
    collect()

test <-
    INPUT_DB %>% 
    tbl('clean_test') %>%
    select(-item_id) %>%
    collect()

test_id <-
    INPUT_DB %>% 
    tbl('test') %>%
    dplyr::select(item_id) %>%
    collect()

label <-
    INPUT_DB %>% 
    tbl('deal_probability') %>%
    select(-item_id) %>%
    collect()
toc()

# Split into Train/Val and convert into Dataset format
# tic('data reorganization')
# val_label <- label[train$activation_date %in% VAL_DATES,]$deal_probability
# train_label <- label[!(train$activation_date %in% VAL_DATES),]$deal_probability

# val_data <- 
#     train %>% 
#     filter(activation_date %in% VAL_DATES) %>%
#     as.matrix

# train_data <- 
#     train %>% 
#     filter(!(activation_date %in% VAL_DATES)) %>%
#     as.matrix

# train_data <- lgb.Dataset(data = train_data,
#                           label = train_label)

# val_data <- lgb.Dataset(data = val_data,
#                         label = val_label)
# toc()

# tic('train model')
# model <- lgb.train(
#     params=TRAIN_PARAMS, 
#     train_data,
#     nrounds=N_ROUNDS,
#     min_data=1,
#     early_stopping_rounds=200,
#     categorical_feature = which(colnames(train_data) %in% CATEGORICAL_FEATURES),
#     valids = list(train = train_data, valid = val_data)
# )
# toc()
### Train full model and predict

train_data <- 
    train %>% 
    as.matrix

train_data <- lgb.Dataset(data = train_data,
                          label = label$deal_probability)
model <- lgb.train(
    params=TRAIN_PARAMS, 
    train_data,
    nrounds=N_ROUNDS,
    categorical_feature = which(colnames(train_data) %in% CATEGORICAL_FEATURES),
    min_data=1
)

deal_probability = predict(model, as.matrix(test))
preds = data_frame(item_id=test_id$item_id, 
                   deal_probability=deal_probability) %>%
        mutate(deal_probability = ifelse(deal_probability>1, 1, deal_probability),
               deal_probability = ifelse(deal_probability<0, 0, deal_probability))

preds %>% write_csv('submission.csv')

# Save output
OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=TRUE)

copy_to(OUTPUT_DB, 
        preds, 
        name=MODEL_ID, 
        temporary=FALSE,
        overwrite = TRUE)


