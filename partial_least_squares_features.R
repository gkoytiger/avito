library(tidyverse)
library(dbplyr)
library(magrittr)
library(tidytext)

DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'
TEXT_DB = 'text_features.sqlite'
OUTPUT_DB = 'engineered_features.sqlite'

WEIGHT_ALPHA = 1e-3 #https://openreview.net/pdf?id=SyK00v5xx

INPUT_DB <-
    file.path(DATA_FOLDER, 'derived', INPUT_DB) %>%
    src_sqlite(create=FALSE)
OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=FALSE)
TEXT_DB <-
    file.path(DATA_FOLDER, 'derived', TEXT_DB) %>%
    src_sqlite(create=FALSE)

words <-
    union_all(
        TEXT_DB %>%
            tbl('clean_train') %>%
            select(item_id, word),
        TEXT_DB %>%
            tbl('clean_test') %>%
            select(item_id, word),
        TEXT_DB %>%
            tbl('clean_train_active') %>%
            select(item_id, word),
        TEXT_DB %>%
            tbl('clean_test_active') %>%
            select(item_id, word),
    ) %>%
    left_join(
        TEXT_DB %>%
            tbl('clean_vocabulary') %>%
            select(word, word_frequency)
    ) %>%
    mutate(sif= WEIGHT_ALPHA / (WEIGHT_ALPHA + word_frequency)) %>%
    collect()
words_matrix = cast_sparse(words, item_id, word, sif)

features <-
    union_all(
        INPUT_DB %>%
            tbl('clean_train'),
        INPUT_DB %>%
            tbl('clean_test'),
        INPUT_DB %>%
            tbl('clean_train_active'),
        INPUT_DB %>%
            tbl('clean_test_active'),
    ) %>% collect()
