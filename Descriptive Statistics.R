
##### PACKAGES #####
library(readr)
library(readxl)
library(dplyr)
library(stringr)
library(tidyr)
library(readr)               # fast I/O
library(dplyr)               # data wrangling
library(stringr)             # tidy string helpers
library(quanteda)            # corpus + tokens
library(quanteda.textstats)  # readability
library(janitor)             # nice frequency / crosstab output
##### Read CSV & Data Manipulation #####
#read CSV with delimiter = ";"
df <- read_delim(
  "CSV/df_clean_all.csv", 
  delim   = ";",        # split on “;” instead of “,”
    col_types = cols(),    # let readr guess each column type
  trim_ws   = TRUE       # trim extra white space
)

#drop abstract_de because not needed
#remove filename
df$`abstracts_de` <- NULL


##### Unique Stuff #####
# Check for unique years
unique_years <- df %>%
  pull(year) %>%       # extract the vector
  unique() %>%          # drop duplicates
  sort()                # sort ascending
print(unique_years)

#unique Kürzel
unique_short_form <- df %>%
  pull(short_form) %>%       # extract the vector
  unique() %>%          # drop duplicates
  sort()                # sort ascending
print(unique_short_form)

#unique Studiengänge
unique_program <- df %>%
  pull(program) %>%       # extract the vector
  unique() %>%          # drop duplicates
  sort()                # sort ascending
print(unique_program)


###### amount of stuff ##### 
count_nonmissing <- function(df, col_name){
  # col_name: character scalar
  sum(!is.na(df[[col_name]]))
}
#returns respective amount
count_nonmissing(df, "topic")
count_nonmissing(df, "abstracts_en")

##### Basic text lengths: words & characters #####
df <- df %>% 
  mutate(
    word_count  = str_count(abstracts_en, "\\S+"),
    char_count  = nchar(abstracts_en)
  )

##### Simple frequency tables (value_counts‑style) #####

freq_year      <- tabyl(df, year)      # or df %>% count(year)
freq_program   <- tabyl(df, short_form)
freq_level     <- tabyl(df, level)
freq_fulltime  <- tabyl(df, fulltime)

# View them:
print(freq_year)
print(freq_program)
print(freq_level)
print(freq_fulltime)

############################################################
## 7.  Cross‑tabs (pivot tables)                           #
############################################################
xtab_year_prog   <- tabyl(df, year,    short_form)   # year × programme
xtab_level_ft    <- tabyl(df, level,   fulltime)  # level × full/part time

# Optional: add totals and nicer formatting
xtab_year_prog <- xtab_year_prog  %>% adorn_totals(where = "rowcol")
xtab_level_ft  <- xtab_level_ft   %>% adorn_totals(where = "rowcol")

print(xtab_year_prog)
print(xtab_level_ft)

############################################################
## 8.  (Optional) write everything to disk                ##
############################################################
write_csv2(freq_year, "yearly_trends.csv")
write_csv2(freq_program, "programme_counts.csv")
write_csv2(xtab_year_prog, "year_vs_program.csv")


