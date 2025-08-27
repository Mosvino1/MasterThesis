library(readr)
library(readxl)
library(dplyr)
library(polyglotr)
library(purrr)
library(pbapply)
library(cld3)
library(stringr)

#identify german and english titles 
# BGE = de
#Bus = en
#Dit = de
#EXP = en
#IVE = en
#IWB = en
#KMU = mixed eher de
#MGT = de
#MGU = de
#MKT = en
#TSM = en
#UNM = de

#read Dataset

df <- read_delim(
  "CSV/df_clean_all.csv", 
  delim   = ";",         # split on “;” instead of “,”
  col_types = cols(),    # let readr guess each column type
  trim_ws   = TRUE # trim extra whitespace
)



safe_gtranslate <- possibly(
  function(txt) {
    google_translate(txt, source_language = "auto", target_language = "en")
  },
  otherwise = NA_character_  # if any connection / quota error
)


df <- df %>%
  mutate(
    topic_en = pbsapply(topic, safe_gtranslate, simplify = "array")
  )

# quick peek
print(df %>% select(topic, topic_en) %>% head(10), width = Inf)

library(dplyr)

df %>%
  distinct(short_form) %>%        # drop duplicates
  pull(short_form) %>%            # extract as vector
  print()                         # show in console

#add original language lables 

df <- df %>% 
  mutate(
    original_language = case_when(
      str_starts(short_form, "BGE") ~ "de",
      str_starts(short_form, "BUS") ~ "en",
      str_starts(short_form, "DIT") ~ "de",
      str_starts(short_form, "EXP") ~ "en",
      str_starts(short_form, "IBE") ~ "en",
      str_starts(short_form, "IWB") ~ "en",
      str_starts(short_form, "KMU") ~ "de",  # eher de
      str_starts(short_form, "MGT") ~ "de",
      str_starts(short_form, "MGU") ~ "de",
      str_starts(short_form, "MKT") ~ "en",
      str_starts(short_form, "TSM") ~ "en",
      str_starts(short_form, "UNM") ~ "de",
      str_starts(short_form, "UF") ~ "de",
      TRUE ~ NA_character_           # fallback if no match
    )
  )

#    write_excel_csv2() uses ";" as delimiter and UTF-8 BOM
readr::write_excel_csv2(df, "df_clean_all_translated.csv")
