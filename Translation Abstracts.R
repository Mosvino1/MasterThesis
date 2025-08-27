library(readr)
library(readxl)
library(dplyr)
library(polyglotr)
library(purrr)

###### 1mport dataset #####
df <- read_delim(
  "df_Clean.csv", 
  delim   = ";",         # split on “;” instead of “,”
  col_types = cols(),    # let readr guess each column type
  trim_ws   = TRUE       # trim extra whitespace
)

print(names(df)) #check headings

#filter out where german is there but no english

to_translate <- df %>%
  filter(
    is.na(english),   # english is NA
    !is.na(german)    # german is not NA
  )

nrow(to_translate)    # how many rows need translation?
glimpse(to_translate) # inspect columns/types
head(to_translate, 5) # peek at the first few


##### Translation Test with polyglotr ########

#Create Test DF
test_df <- to_translate %>% 
  slice(1:15) %>% 
  # make sure empty strings are treated as NA
  mutate(
    german  = na_if(german,  ""),
    english = na_if(english, "")
  )

# Translate German→English for rows where english is NA
test_df_translated <- test_df %>%
  mutate(
    english = map_chr(
      german,
      ~ google_translate_long_text(
        text              = .x,
        target_language   = "en",
        source_language   = "de",
        chunk_size        = 1000,
        preserve_newlines = FALSE
      )
    )
  )

# 4. View result
View(test_df_translated)

###### Entire DF #######
# 1. Build a “safe” version that returns NA on error
safe_gtlt <- possibly(
  google_translate_long_text,
  otherwise    = NA_character_
)

# 2. Translate all German abstracts in one go
to_translate <- to_translate %>%
  mutate(
    english = map_chr(
      german,
      ~ {
        # (Optional) pause between calls to be nice to the API
        Sys.sleep(0.2)  
        
        # call the safe translator
        safe_gtlt(
          text               = .x,
          target_language    = "en",
          source_language    = "de",
          chunk_size         = 1000,
          preserve_newlines  = FALSE
        )
      }
    )
  )

# 3. Quick checks
sum(is.na(to_translate$english))    # how many still failed?
head(to_translate, 5)               # peek at translations
View(to_translate)
#206 abstracts translated


# 1. Build the same logical index you used to create to_translate
idx <- is.na(df$english) & !is.na(df$german)

# 2. Copy the newly translated text back into df
df$english[idx] <- to_translate$english

# 3. Write out a semicolon-delimited UTF-8-safe CSV for Excel
#    write_excel_csv2() uses ";" as delimiter and UTF-8 BOM
readr::write_excel_csv2(df, "df_Clean_inkl_translated.csv")
