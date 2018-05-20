library(tidyverse)
library(dbplyr)
library(magrittr)
library(broom)
library(foreach)
library(iterators)

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

QUANTILES = seq(0, 1, 0.1)

### Word usage features
#Remember: The word counts are from the combined dataset including actives
word_frequency_stats <-
    union_all(
        TEXT_DB %>%
            tbl('clean_train') %>%
            select(item_id, word),
        TEXT_DB %>%
            tbl('clean_test') %>%
            select(item_id, word)
    ) %>%
    inner_join(
        TEXT_DB %>%
        tbl('vocabulary') %>%
        select(word, word_frequency)
    ) %>%
    collect()

word_frequency_quantiles <-
    word_frequency_stats %>%
    group_by(item_id) %>%
    summarise(quantiles = list(str_c('Q', as.integer(QUANTILES*100))),
              word_frequency = list(quantile(word_frequency, QUANTILES, na.rm=TRUE))) %>% 
    unnest %>%
    spread(quantiles,word_frequency)

copy_to(OUTPUT_DB, word_frequency_quantiles, unique_indexes=list('item_id'))

### Category price quantiles
GROUP_CATS = quos(parent_category_name, category_name, param_1, user_type)
category_prices <-
    union_all(
        INPUT_DB %>%
            tbl('train') %>%
            select(!!!GROUP_CATS, log_price),
        INPUT_DB %>%
            tbl('test') %>%
            select(!!!GROUP_CATS, log_price),
        INPUT_DB %>%
            tbl('train_active') %>%
            select(!!!GROUP_CATS, log_price),
        INPUT_DB %>%
            tbl('test_active') %>%
            select(!!!GROUP_CATS, log_price)
    ) %>%
    collect() 

price_quantiles <-
    category_prices %>%
    group_by(!!!GROUP_CATS) %>%
    filter(log_price > 0) %>%
    summarise(quantiles = list(str_c('Q', as.integer(QUANTILES*100))),
              log_price = list(quantile(log_price, QUANTILES, na.rm=TRUE))) %>% 
    unnest %>%
    spread(quantiles,log_price) %>%
    select(-Q0)
copy_to(OUTPUT_DB, price_quantiles, overwrite=TRUE,
        indexes=list('parent_category_name', 'category_name', 'param_1', 'user_type'))

### Category post count
# Identifies how popular certain categories are, trying to estimate post competition
GROUP_CATS = quos(parent_category_name, category_name, activation_date, city)
num_posts <-
    union_all(
        INPUT_DB %>%
            tbl('train') %>%
            select(!!!GROUP_CATS),
        INPUT_DB %>%
            tbl('test') %>%
            select(!!!GROUP_CATS),
        INPUT_DB %>%
            tbl('train_active') %>%
            select(!!!GROUP_CATS),
        INPUT_DB %>%
            tbl('test_active') %>%
            select(!!!GROUP_CATS)
    ) %>%
    count(!!!GROUP_CATS) %>%
    collect()

copy_to(OUTPUT_DB, num_posts, overwrite=TRUE,
        indexes=list('parent_category_name', 'category_name', 'activation_date', 'city'))

### Embed title and description
text_data <-
    union_all(
        TEXT_DB %>%
            tbl('clean_train') %>%
            select(item_id, field, word),
        TEXT_DB %>%
            tbl('clean_test') %>%
            select(item_id, field, word)
    ) %>%
    inner_join(
        TEXT_DB %>%
        tbl('clean_vocabulary') %>%
        select(word)
    ) %>%
    collect() 


word_vecs <-
    TEXT_DB %>%
    tbl('word_vecs')%>%
    inner_join(
        TEXT_DB %>%
        tbl('clean_vocabulary') %>%
        select(word, sif)
    ) %>%
    collect()

text_groups <- text_data %>% nest(item_id, field)
summarize_words <- function(df){
    df %>%
        left_join(word_vecs, by = 'word') %>%
        select(-word) %>%
        summarize_all(funs(weighted.mean(., sif))) %>%
        select(-sif) %>%
        as.numeric
}
text_groups['embedding'] = map(text_groups['data'][[1]], summarize_words)
