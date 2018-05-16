library(tidyverse)
library(dbplyr)
library(tidytext)
library(foreach)
library(magrittr)
library(doParallel)
registerDoParallel(4)

### Globals
TEST_MODE=FALSE
DATA_FOLDER = '~/.kaggle/competitions/avito-demand-prediction/'
TEXT_FILES = c('test.csv',
                'train.csv',
                'test_active.csv',
                'train_active.csv')
OUTPUT_DB = 'baseline_features.sqlite'

### Inits
if(TEST_MODE){
    n_max=500
}else{
    n_max=Inf
}

OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=TRUE)

city_locations = read_csv(file.path(DATA_FOLDER, 'city_locations.csv'))

foreach(text_file = TEXT_FILES) %dopar%{
    text <-
        file.path(DATA_FOLDER, text_file) %>%
            read_csv(n_max=n_max) 
    
    if(text_file == 'train.csv'){
        deal_probability <- text %>% select(item_id, deal_probability)
        copy_to(OUTPUT_DB, 
                deal_probability, 
                temporary=FALSE,
                overwrite = TRUE, 
                unique_indexes=list('item_id'))
        
        text %<>% select(-deal_probability)
    } 
    
    text %<>% 
        mutate(
            no_img = is.na(image) %>% as.integer(),
            no_dsc = is.na(description) %>% as.integer(),
            no_p1 = is.na(param_1) %>% as.integer(), 
            no_p2 = is.na(param_2) %>% as.integer(), 
            no_p3 = is.na(param_3) %>% as.integer(),
            titl_len = str_length(title),
            titl_len_log = log1p(titl_len),
            desc_len = str_length(description),
            desc_len_log = log1p(desc_len),
            titl_cap = str_count(title, "[A-ZА-Я]"),
            titl_cap_log = log1p(titl_cap),
            desc_cap = str_count(description, "[A-ZА-Я]"),
            desc_cap_log = log1p(desc_cap),
            titl_pun = str_count(title, "[[:punct:]]"),
            titl_pun_log = log1p(titl_pun),
            desc_pun = str_count(description, "[[:punct:]]"),
            desc_pun_log = log1p(desc_pun),
            titl_dig = str_count(title, "[[:digit:]]"),
            titl_dig_log = log1p(titl_dig),
            desc_dig = str_count(description, "[[:digit:]]"),
            desc_dig_log = log1p(desc_dig),
            log_price = log1p(price),
            location = str_c(city, ', ', region)
        ) %>%
        left_join(city_locations) %>%
        select(-title, -description, -location) %>%
        distinct(item_id)
    
    indexes = text %>% select_if(is.character) %>% select(-item_id) %>% colnames

    copy_to(OUTPUT_DB, 
            text, 
            temporary=FALSE,
            overwrite = TRUE, 
            name = str_replace(text_file,'.csv',''),
            indexes=as.list(indexes),
            unique_indexes=list('item_id'))
    NULL
}
 
