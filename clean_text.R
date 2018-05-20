library(tidyverse)
library(dbplyr)
library(tidytext)
library(foreach)
library(magrittr)
library(doParallel)
registerDoParallel(4)
library(stopwords)

### Globals
DATA_FOLDER = '/home/greg/.kaggle/competitions/avito-demand-prediction'
TEXT_FILES = c('test.csv',
                'train.csv',
                'test_active.csv',
                'train_active.csv')

OUTPUT_DB = 'text_features.sqlite'
OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=FALSE)

MIN_WORD_LEN = 4
WORD_VECTORS = file.path(DATA_FOLDER, 'wiki.ru')
ALPHA_PARAM = 1e-3

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

### Clean text
clean_vocabulary <-
    OUTPUT_DB %>%
    tbl('vocabulary') %>%
    arrange(-word_frequency) %>%
    filter(n > 100) %>%
    anti_join(OUTPUT_DB %>% tbl('stopwords')) %>%
    inner_join(OUTPUT_DB %>% tbl('test') %>% distinct(word)) %>%
    collect() %>%
    mutate(word_length = str_length(word)) %>%
    filter(word_length >= MIN_WORD_LEN) %>%
    select(-word_length) %>%
    mutate(sif = ALPHA_PARAM / (ALPHA_PARAM + word_frequency))

copy_to(OUTPUT_DB, clean_vocabulary, temporary=FALSE, unique_indexes=list('word'), overwrite=TRUE)

### Calculated fasttext embeding using python

### Simplify vocab by clustering words
word_vecs <-
    OUTPUT_DB %>%
    tbl('word_vecs') %>%
    collect()
hc <- hclust(as.dist(1-cor(word_vecs %>% select(-word) %>% t())))
meta_word = cutree(hc, k = 1000)
vocab_map = bind_cols(word_vecs['word'],data_frame(meta_word))
total_count = sum(clean_vocabulary$n)
simplified_vocabulary <-
    vocab_map %>%
    left_join(
        clean_vocabulary %>% select(word, n)
    ) %>%
    group_by(meta_word) %>%
    summarize(
        word_frequency = sum(n)/total_count,
        n = sum(n)
    ) %>%
    mutate(sif = ALPHA_PARAM / (ALPHA_PARAM + word_frequency)) %>%
    left_join(vocab_map) %>%
    left_join(clean_vocabulary %>% select(word))

copy_to(OUTPUT_DB, simplified_vocabulary, temporary=FALSE, unique_indexes=list('word'), overwrite=TRUE)


clean_train <-
    OUTPUT_DB %>% 
    tbl('train') %>%
    inner_join(
        OUTPUT_DB %>%
        tbl('simplified_vocabulary') %>%
        select(word, meta_word)
     ) %>%
    collect()
copy_to(OUTPUT_DB, clean_train, temporary=FALSE, overwrite=TRUE,
        indexes=list('item_id', 'field', 'word', 'meta_word'))

clean_test <-
    OUTPUT_DB %>% 
    tbl('test') %>%
    inner_join(
        OUTPUT_DB %>%
        tbl('simplified_vocabulary') %>%
        select(word, meta_word)
     ) %>%
    collect()
copy_to(OUTPUT_DB, clean_test, temporary=FALSE, overwrite=TRUE,
        indexes=list('item_id', 'field', 'word', 'meta_word'))

clean_train_active <-
    OUTPUT_DB %>% 
    tbl('train_active') %>%
    inner_join(
        OUTPUT_DB %>%
        tbl('simplified_vocabulary') %>%
        select(word, meta_word)
     ) %>%
    collect()
copy_to(OUTPUT_DB, clean_train_active, temporary=FALSE, overwrite=TRUE,
        indexes=list('item_id', 'field', 'word', 'meta_word'))

clean_test_active <-
    OUTPUT_DB %>% 
    tbl('test_active') %>%
    inner_join(
        OUTPUT_DB %>%
        tbl('simplified_vocabulary') %>%
        select(word, meta_word)
     ) %>%
    collect()
copy_to(OUTPUT_DB, clean_test_active, temporary=FALSE, overwrite=TRUE,
        indexes=list('item_id', 'field', 'word', 'meta_word'))
