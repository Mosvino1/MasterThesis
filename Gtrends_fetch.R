library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(purrr)
library(gtrendsR)

# ------------------ CONFIG ------------------
FILE  <- "topic_year_aggregation_labeled 1.1.xlsx"
SHEET <- "keywords_per_topic_gtrends"
GEO   <- ""                  # "" = worldwide, "AT" = Austria
YEAR_START <- 2015
YEAR_END   <- 2024
TIME <- sprintf("%d-01-01 %d-12-31", YEAR_START, YEAR_END)
# --------------------------------------------

# Read Excel and pick first 5 keyword columns
raw <- readxl::read_xlsx(FILE, sheet = SHEET)

kw_cols <- names(raw)[grepl("(?i)^keyword\\s*\\d+$", names(raw))]
kw_cols <- kw_cols[order(readr::parse_number(kw_cols))]
kw_cols <- kw_cols[seq_len(min(5, length(kw_cols)))]

dat <- raw |>
  dplyr::select(topic, dplyr::all_of(kw_cols))

n_topics <- nrow(dat)
pb <- utils::txtProgressBar(min = 0, max = n_topics, style = 3)
res_list <- vector("list", n_topics)

for (i in seq_len(n_topics)) {
  row_i <- dat[i, , drop = FALSE]
  # Extract and clean keywords (drop blanks/NA, trim, de-duplicate)
  kws <- as.character(unlist(row_i[1, kw_cols], use.names = FALSE))
  kws <- trimws(kws)
  kws <- kws[nzchar(kws) & !is.na(kws)]
  kws <- unique(kws)
  
  if (length(kws) == 0) {
    res_df <- tibble::tibble()
  } else {
    out <- try(
      gtrendsR::gtrends(
        keyword = kws,
        geo = GEO,
        time = TIME,
        onlyInterest = TRUE,
        low_search_volume = TRUE
      ),
      silent = TRUE
    )
    
    if (inherits(out, "try-error") || is.null(out$interest_over_time)) {
      res_df <- tibble::tibble()
    } else {
      # Normalize types to avoid bind_rows conflicts
      res_df <- out$interest_over_time |>
        dplyr::select(dplyr::any_of(c("date","hits","keyword","geo","gprop"))) |>
        dplyr::mutate(
          date    = as.Date(.data$date),
          hits    = as.character(.data$hits),     # force character; "<1" shows up here
          keyword = as.character(.data$keyword),
          geo     = as.character(.data$geo),
          gprop   = as.character(.data$gprop),
          topic   = row_i$topic[[1]]
        )
    }
  }
  
  res_list[[i]] <- res_df
  utils::setTxtProgressBar(pb, i)
  # Sys.sleep(0.2)  # (optional) be gentler to the endpoint
}
close(pb)

# Combine all topics' results
iot_all <- dplyr::bind_rows(res_list)
if (nrow(iot_all) == 0) stop("No data returned. Check GEO/TIME or keywords.")

# Yearly mean per keyword, then average across keywords within each topic
iot_clean <- iot_all |>
  dplyr::mutate(
    year     = lubridate::year(.data$date),
    hits_num = dplyr::if_else(.data$hits == "<1", 0, suppressWarnings(as.numeric(.data$hits)))
  ) |>
  dplyr::filter(dplyr::between(.data$year, YEAR_START, YEAR_END), !is.na(.data$hits_num))

kw_year <- iot_clean |>
  dplyr::group_by(.data$topic, .data$keyword, .data$year) |>
  dplyr::summarise(mean_kw_year = mean(.data$hits_num, na.rm = TRUE), .groups = "drop")

topic_year <- kw_year |>
  dplyr::group_by(.data$topic, .data$year) |>
  dplyr::summarise(gt_mean = mean(.data$mean_kw_year, na.rm = TRUE), .groups = "drop")

# Wide table: topic × 2015…2024
wide <- topic_year |>
  dplyr::mutate(year = as.character(.data$year)) |>
  tidyr::pivot_wider(
    names_from = .data$year,
    values_from = .data$gt_mean,
    values_fill = 0
  ) |>
  dplyr::arrange(suppressWarnings(as.numeric(.data$topic)), .by_group = FALSE)

# Preview
print(wide, n = 30)

# --- Save (pick one) ---
# CSV with semicolons (EU style):
write.csv2(wide, "gtrends_topic_yearly_mean_2015_2024.csv", row.names = FALSE)

# Excel (if openxlsx is installed):
# openxlsx::write.xlsx(wide, "gtrends_topic_yearly_mean_2015_2024.xlsx", asTable = TRUE)
