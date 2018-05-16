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
OUTPUT_DB = 'price_quantiles.sqlite'

### Inits
if(TEST_MODE){
    n_max=500
}else{
    n_max=Inf
}

OUTPUT_DB <-
    file.path(DATA_FOLDER, 'derived', OUTPUT_DB) %>%
    src_sqlite(create=TRUE)