library(tidyverse)
library(dbplyr)
library(magrittr)
library(broom)
library(foreach)
library(iterators)
library(tidytext)
library(irlba)
library(textir)

### Inits
DATA_FOLDER = '/home/greg/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'
TEXT_DB = 'text_features.sqlite'
OUTPUT_DB = 'engineered_features.sqlite'

INPUT_DB <-
    file.path(DATA_FOLDER, 'derived', INPUT_DB) %>%
    src_sqlite(create=FALSE)
OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=FALSE)
TEXT_DB <-
    file.path(DATA_FOLDER, 'derived', TEXT_DB) %>%
    src_sqlite(create=FALSE)


### Unsupervised PCA with components estimated on all data
N_TOPICS=50
text_features <-
    union_all(
        TEXT_DB %>%
            tbl('clean_test') %>%
            select(item_id, meta_word),
        TEXT_DB %>%
                tbl('clean_train') %>%
                select(item_id, meta_word),
        TEXT_DB %>%
                tbl('clean_train_active') %>%
                select(item_id, meta_word),
        TEXT_DB %>%
                tbl('clean_test_active') %>%
                select(item_id, meta_word),
    ) %>%
    inner_join(
        TEXT_DB %>%
                tbl('simplified_vocabulary') %>%
                select(meta_word, sif)
    ) %>%
    group_by(item_id, meta_word) %>%
    summarize(weight = sum(sif)) %>%
    collect()

item_ids <-
    union_all(
        TEXT_DB %>%
            tbl('clean_test') %>%
            select(item_id),
        TEXT_DB %>%
                tbl('clean_train') %>%
                select(item_id)
    ) %>%
    collect()

text_matrix <- cast_sparse(text_features, item_id, meta_word, weight)
saveRDS(text_matrix, file.path(DATA_FOLDER, 'derived', 'meta_text_matrix.rds'))

text_svd <- svdr(text_matrix, k=N_TOPICS)
topics = bind_cols(as_data_frame(rownames(text_matrix)), as_data_frame(text_svd$u))

#Join test and train item ids
topics %<>% rename(item_id=value) %>% semi_join(item_ids)

copy_to(OUTPUT_DB, topics, overwrite=TRUE, name='pca_topics',
        unique_indexes=list('item_id'))


### Price topics using plsr
prices <-
    union_all(
        INPUT_DB %>%
            tbl('clean_test') %>%
            select(item_id, log_price),
        INPUT_DB %>%
                tbl('clean_train') %>%
                select(item_id, log_price),
        INPUT_DB %>%
                tbl('clean_train_active') %>%
                select(item_id, log_price),
        INPUT_DB %>%
                tbl('clean_test_active') %>%
                select(item_id, log_price),
    ) %>%
    collect()
prices$log_price[is.na(prices$log_price)]=0

prices <- 
    data_frame(item_id=rownames(text_matrix)) %>%
    left_join(prices)

price_topics = pls(x=text_matrix, y=prices$log_price, K=N_TOPICS)
price_topics = bind_cols(data_frame(item_id=rownames(text_matrix)), 
                         as_data_frame(price_topics$directions)) %>%
               semi_join(item_ids)

copy_to(OUTPUT_DB, price_topics, overwrite=TRUE, name='price_topics',
        unique_indexes=list('item_id'))

price_topics = pls(x=text_matrix, y=prices$log_price, K=N_TOPICS, scale=FALSE)
price_topics = bind_cols(data_frame(item_id=rownames(text_matrix)), 
                         as_data_frame(price_topics$directions)) %>%
               semi_join(item_ids)

copy_to(OUTPUT_DB, price_topics, overwrite=TRUE, name='price_topics_unscaled',
        unique_indexes=list('item_id'))