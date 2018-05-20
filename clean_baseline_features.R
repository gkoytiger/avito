library(tidyverse)
library(dbplyr)
library(foreach)
library(magrittr)
library(lightgbm)
library(Matrix)
library(tictoc)

DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'

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

# File I/O
tic('Loading Data')
INPUT_DB <-
    file.path(DATA_FOLDER, 'derived', INPUT_DB) %>%
    src_sqlite(create=FALSE)

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

# Preparing rules on test to ignore features that irrelevant to test time
tic('prepare rules')
test_data <- lgb.prepare_rules(test)
full_train_data <- lgb.prepare_rules(data = train, rules = test_data$rules)$data
toc()
saveRDS(test_data$rules, file.path(DATA_FOLDER, 'derived', 'test_rules.rds'))

# Save cleaned datasets
clean_train = bind_cols(train_id, full_train_data)
clean_test = bind_cols(test_id, test_data$data)

copy_to(INPUT_DB, clean_train, temporary=FALSE,
        indexes=list(CATEGORICAL_FEATURES), 
        unique_indexes=list('item_id'))

copy_to(INPUT_DB, clean_test, temporary=FALSE,
        indexes=list(CATEGORICAL_FEATURES), 
        unique_indexes=list('item_id'))

### Perform for actives
test_data=list()
test_data$rules = readRDS(file.path(DATA_FOLDER, 'derived', 'test_rules.rds'))

train_active <- 
    INPUT_DB %>% 
    tbl('train_active') %>%
    dplyr::select(-title, -description, -user_id) %>%
    collect()

clean_train_active <- lgb.prepare_rules(data = train_active %>% select(-item_id),
                                        rules = test_data$rules)$data
clean_train_active = bind_cols(train_active['item_id'], clean_train_active)

copy_to(INPUT_DB, clean_train_active, temporary=FALSE,
        indexes=list(CATEGORICAL_FEATURES), 
        unique_indexes=list('item_id'))

tic('process test active')
test_active <- 
    INPUT_DB %>% 
    tbl('test_active') %>%
    dplyr::select(-title, -description, -user_id) %>%
    collect() 
clean_test_active <- lgb.prepare_rules(data = test_active %>% select(-item_id),
                                       rules = test_data$rules)$data
clean_test_active = bind_cols(test_active['item_id'], clean_train_active)
                                      
copy_to(INPUT_DB, test_active, temporary=FALSE,
        indexes=list(CATEGORICAL_FEATURES), 
        unique_indexes=list('item_id'))
toc()