library(tidyverse)
library(dbplyr)

setwd('~/Desktop/avito/avito')

TEST_MODE=FALSE
OUTPUT_FILE = '../data/derived/other_deals.sqlite'


if(TEST_MODE){
  active=read_csv('../data/train_active.csv.zip', n_max=50000)
}else{
  active = 
    bind_rows(
      read_csv('../data/train_active.csv.zip'),
      read_csv('../data/test_active.csv.zip')
    )
}

other_deals <-
  active %>%
  mutate(
    price=ifelse(is.nan(price), -9, price),
    price=log10(price+10)) %>%
  group_by(region, city, parent_category_name, category_name) %>%
  mutate(
    other_deals_count = n(),
    other_deals_price_mean = mean(price),
    other_deals_price_q0=quantile(price, probs=0, na.rm = TRUE),
    other_deals_price_q25=quantile(price, probs=0.25, na.rm = TRUE),
    other_deals_price_q50=quantile(price, probs=0.5, na.rm = TRUE),
    other_deals_price_q75=quantile(price, probs=0.75, na.rm = TRUE),
    other_deals_price_q100=quantile(price, probs=1, na.rm = TRUE)
  )

output_db = src_sqlite(OUTPUT_FILE, create=TRUE)
copy_to(output_db, other_deals, temporary=FALSE, overwrite=TRUE)
