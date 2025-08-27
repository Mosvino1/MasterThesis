

# ------------ Config (paths & sheet) ------------
GTR_PATH <- "gtrends_topic_yearly_mean_2015_2024.csv"  # semicolon-separated
TOPIC_PATH <- "Topic_percent_yearly fixed.xlsx"        # has assigned_topic, CustomName, 2015..2024
TOPIC_SHEET <- 1                                       # or "Tabelle1"

# ------------ Libraries ------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(scales)
})

# ------------ Helpers for reading / shaping ------------
.as_percent <- function(x) {
  if (all(is.na(x))) return(x)
  mx <- suppressWarnings(max(x, na.rm = TRUE))
  if (!is.finite(mx)) return(x)
  if (mx <= 1) x * 100 else x
}

.to_long_years <- function(df) {
  year_cols <- names(df)[grepl("^(2015|2016|2017|2018|2019|2020|2021|2022|2023|2024)$", names(df))]
  df %>%
    pivot_longer(all_of(year_cols), names_to = "year", values_to = "value") %>%
    mutate(
      year  = as.integer(year),
      value = readr::parse_number(as.character(value))
    )
}

# CSV reader (semicolon + possible comma decimals)
.read_gtrends_csv <- function(path) {
  # Read as character to be safe, we will parse later
  readr::read_delim(path,
                    delim = ";",
                    locale = readr::locale(decimal_mark = ","),
                    show_col_types = FALSE) %>%
    mutate(topic = as.character(topic)) %>%
    mutate(topic = stringr::str_squish(topic))
}

.read_topic_xlsx <- function(path, sheet = 1) {
  df <- readxl::read_excel(path, sheet = sheet)
  if ("assigned_topic" %in% names(df)) {
    names(df)[names(df) == "assigned_topic"] <- "topic"
  }
  if (!("topic" %in% names(df))) {
    # assume first col is topic id
    df <- df %>% mutate(across(1, as.character))
    names(df)[1] <- "topic"
  }
  df %>%
    mutate(topic = as.character(topic)) %>%
    mutate(topic = stringr::str_squish(topic))
}

# ------------ LOAD DATA (done at the beginning) ------------
if (!file.exists(GTR_PATH)) stop("Google Trends CSV not found at: ", GTR_PATH)
if (!file.exists(TOPIC_PATH)) stop("Topic percent XLSX not found at: ", TOPIC_PATH)

G_RAW <- .read_gtrends_csv(GTR_PATH)           # wide: topic, 2015..2024
T_RAW <- .read_topic_xlsx(TOPIC_PATH, TOPIC_SHEET)  # wide: topic, CustomName, 2015..2024

if (nrow(T_RAW) == 0) stop("The topic XLSX sheet appears to be empty.")

# Long formats
G_LONG <- .to_long_years(G_RAW) %>%
  rename(gtrends = value) %>%
  arrange(topic, year)

T_LONG <- .to_long_years(T_RAW) %>%
  rename(topic_pct_raw = value) %>%
  mutate(topic_pct = .as_percent(topic_pct_raw)) %>%
  arrange(topic, year)

# Topic lookup (ID <-> name)
TOPIC_LOOKUP <- T_RAW %>%
  select(topic, CustomName) %>%
  mutate(CustomName = ifelse(is.na(CustomName) | CustomName == "", topic, CustomName))

# ------------ Convenience: list available topics ------------
list_topics <- function() {
  if ("CustomName" %in% names(TOPIC_LOOKUP)) {
    TOPIC_LOOKUP %>%
      mutate(label = paste0(topic, " - ", CustomName)) %>%
      arrange(as.numeric(topic)) %>%
      pull(label)
  } else {
    G_RAW %>% pull(topic) %>% unique() %>% sort()
  }
}

# ------------  resolve topic input ------------
.resolve_topic <- function(x_topic) {
  # Accept numeric ID (as char) or CustomName (case-insensitive exact match)
  x_chr <- as.character(x_topic)
  
  # 1) Try by CustomName exact (case-insensitive)
  if ("CustomName" %in% names(TOPIC_LOOKUP)) {
    hit <- TOPIC_LOOKUP %>% filter(tolower(CustomName) == tolower(x_chr))
    if (nrow(hit) == 1) {
      return(list(id = as.character(hit$topic[1]),
                  name = as.character(hit$CustomName[1])))
    }
  }
  # 2) Try by topic id exact
  if (x_chr %in% G_LONG$topic) {
    nm <- TOPIC_LOOKUP %>% filter(topic == x_chr) %>% pull(CustomName)
    nm <- ifelse(length(nm) == 1, nm, x_chr)
    return(list(id = x_chr, name = nm))
  }
  
  # 3) Offer candidates (partial)
  cands <- c()
  if ("CustomName" %in% names(TOPIC_LOOKUP)) {
    cands <- c(cands, TOPIC_LOOKUP$CustomName[stringr::str_detect(
      TOPIC_LOOKUP$CustomName, stringr::fixed(x_chr, ignore_case = TRUE))])
  }
  cands <- c(cands, unique(G_LONG$topic[stringr::str_detect(
    G_LONG$topic, stringr::fixed(x_chr, ignore_case = TRUE))]))
  
  stop("Topic not found: '", x_topic, "'. Candidates: ",
       paste(head(unique(cands), 10), collapse = "; "),
       ifelse(length(unique(cands)) > 10, " …", ""))
}

# ----------- plotting ---------

plot_topic_dual <- function(topic,
                            save_path = NULL,
                            display   = c("both", "gtrends", "topic")) {
  display <- match.arg(display)
  
  g_col <- "#0072B2"   # Google Trends blue
  t_col <- "#111111"   # Topic share black
  
  # Resolve topic to id + nice name (uses objects loaded at the top)
  sel <- .resolve_topic(topic)
  sel_id <- sel$id
  sel_name <- sel$name
  
  g_sel <- G_LONG %>% dplyr::filter(as.character(topic) == sel_id)
  t_sel <- T_LONG %>% dplyr::filter(as.character(topic) == sel_id)
  
  if (nrow(g_sel) == 0) stop("No Google Trends data for topic id: ", sel_id)
  if (nrow(t_sel) == 0) stop("No Topic Share data for topic id: ", sel_id)
  
  df <- dplyr::full_join(g_sel, t_sel, by = c("topic", "year")) %>% dplyr::arrange(year)
  
  # ----- dynamic left-axis bounds for Topic share (%)
  t_min <- suppressWarnings(min(df$topic_pct, na.rm = TRUE))
  t_max <- suppressWarnings(max(df$topic_pct, na.rm = TRUE))
  span  <- t_max - t_min
  pad   <- if (is.finite(span) && span > 0) span * 0.15 else max(0.1, t_max * 0.15)
  y_min <- max(0, t_min - pad)
  y_max <- t_max + pad
  if (!is.finite(y_min) || !is.finite(y_max) || y_min >= y_max) {
    # fallback if series is flat or NA
    y_min <- 0
    y_max <- max(1, t_max * 1.25)
  }
  
  # ----- map GTrends [0,100] to the left axis [y_min, y_max]
  a <- (y_max - y_min) / 100
  b <- y_min
  df$gtrends_on_left <- a * df$gtrends + b  # for plotting on primary axis
  inv <- function(y) (y - b) / a            # inverse for sec.axis labels (0..100)
  
  if (display == "gtrends") {
    # Show only Google Trends (single left axis, fixed 0–100)
    p <- ggplot(df, aes(x = year, y = gtrends)) +
      geom_line(linetype = "dotted", linewidth = 1, color = g_col, na.rm = TRUE) +
      geom_point(color = g_col, size = 1.8, na.rm = TRUE) +
      scale_x_continuous(breaks = sort(unique(df$year))) +
      scale_y_continuous(name = "Google Trends (0–100)", limits = c(0, 100), breaks = seq(0, 100, 25)) +
      labs(title = paste0(sel_name, " — Google Trends"),
           subtitle = "Years 2015–2024", x = NULL) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor = element_blank(),
        axis.title.y = element_text(color = g_col),
        axis.text.y  = element_text(color = g_col)
      )
    
  } else if (display == "topic") {
    # Show only Topic share (single left axis, dynamic bounds)
    p <- ggplot(df, aes(x = year, y = topic_pct)) +
      geom_line(linewidth = 1, color = t_col, na.rm = TRUE) +
      geom_point(size = 1.8, color = t_col, na.rm = TRUE) +
      scale_x_continuous(breaks = sort(unique(df$year))) +
      scale_y_continuous(name = "Topic share (%)",
                         limits = c(y_min, y_max),
                         breaks = scales::breaks_pretty(n = 6)) +
      labs(title = paste0(sel_name, " — Topic share"),
           subtitle = "Years 2015–2024", x = NULL) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor = element_blank(),
        axis.title.y = element_text(color = t_col),
        axis.text.y  = element_text(color = t_col)
      )
    
  } else {
    # BOTH: Topic share LEFT (black), Google Trends RIGHT (blue, fixed 0–100)
    p <- ggplot(df, aes(x = year)) +
      # Topic share on the LEFT axis (primary)
      geom_line(aes(y = topic_pct), linewidth = 1, color = t_col, na.rm = TRUE) +
      geom_point(aes(y = topic_pct), size = 1.8, color = t_col, na.rm = TRUE) +
      # Google Trends mapped onto LEFT scale but labelled on RIGHT axis
      geom_line(aes(y = gtrends_on_left), linetype = "dotted", linewidth = 1, color = g_col, na.rm = TRUE) +
      geom_point(aes(y = gtrends_on_left), size = 1.8, color = g_col, na.rm = TRUE) +
      scale_x_continuous(breaks = sort(unique(df$year))) +
      scale_y_continuous(
        name    = "Topic share (%)",
        limits  = c(y_min, y_max),
        breaks  = scales::breaks_pretty(n = 6),
        sec.axis = sec_axis(trans = inv,
                            name = "Google Trends (0–100)",
                            breaks = seq(0, 100, 25))
      ) +
      labs(title = paste0(sel_name, " — Topic share vs. Google Trends"),
           subtitle = "Years 2015–2024", x = NULL) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor = element_blank(),
        # LEFT (Topic share) in black
        axis.title.y = element_text(color = t_col),
        axis.text.y  = element_text(color = t_col),
        # RIGHT (GTrends) in blue
        axis.title.y.right = element_text(color = g_col),
        axis.text.y.right  = element_text(color = g_col)
      )
  }
  
  if (!is.null(save_path)) {
    ggsave(filename = save_path, plot = p, width = 9, height = 5.2, dpi = 300)
  }
  p
}

# ----------- Plot -----------
list_topics()
plot_topic_dual("Artificial Intelligence", display = "both")
plot_topic_dual("22", display = "both")   # by numeric ID as character
# plot_topic_dual("Artificial Intelligence", display = "topic", save_path = "ai_topic_only.png")
