library(tidyverse)
library(dbplyr)
library(tidytext)
library(foreach)
library(magrittr)
library(doParallel)
registerDoParallel(4)
library(stopwords)

### Globals
DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
TEXT_FILES = c('test.csv',
                'train.csv',
                'test_active.csv',
                'train_active.csv')

OUTPUT_DB = 'text_features.sqlite'
OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=TRUE)

TEST_MODE=FALSE
if(TEST_MODE){
    n_max=500
}else{
    n_max=Inf
}

### Functions
clean_text <- function(txt){
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") 
}

### Main
foreach(text_file = TEXT_FILES) %dopar%{
    text <-
        file.path(DATA_FOLDER, text_file) %>%
            read_csv(n_max=n_max) %>%
            select(item_id, title, description) %>%
            gather(field, txt, title, description) %>%
            mutate(txt = clean_text(txt)) %>%
            unnest_tokens(word, txt, to_lower=FALSE) 
    
    copy_to(OUTPUT_DB, text, temporary=FALSE,
            overwrite = TRUE, name = str_replace(text_file,'.csv',''),
            indexes = list('item_id', 'field', 'word'))
    NULL
}

vocabulary <-
    union_all(
        OUTPUT_DB %>%
            tbl('test') %>%
            select(word),
        OUTPUT_DB %>%
                tbl('train') %>%
                select(word),
        OUTPUT_DB %>%
                tbl('train_active') %>%
                select(word),
        OUTPUT_DB %>%
                tbl('test_active') %>%
                select(word),
    ) %>%
    count(word) %>%
    collect() %>%
    mutate(word_frequency = n/sum(n))

copy_to(OUTPUT_DB, vocabulary, temporary=FALSE,
        overwrite = TRUE,
        indexes = list('word'))

stopwords = data_frame(word=stopwords('ru'))
copy_to(OUTPUT_DB, stopwords, temporary=FALSE,
        overwrite = TRUE,
        indexes = list('word'))