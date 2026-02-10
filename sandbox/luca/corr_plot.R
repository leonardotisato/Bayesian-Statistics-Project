library(GGally)
library(dplyr)

data <- read.csv("sandbox/data/add_data/df_corr.csv")

ecec_vars <-  c('ecec_diffusion',
                'ecec_participation',
                'other_percapita_public_expenditure',
                'other_percapita_user_contrib',
                'private_coverage',
                'public_coverage')

corr_data <-  data %>% select(-all_of(ecec_vars))
num_cols <-  corr_data %>% select(where(is.numeric)) %>% names()

df_plot <- corr_data %>%
  select(all_of(num_cols), macro_area) %>%
  mutate(macro_area = as.factor(macro_area))

p <- ggpairs(
  df_plot,
  columns = num_cols,                 # only numeric columns are plotted
  mapping = aes(colour = macro_area),  # color by macro_area
  upper = list(continuous = "cor"),    # correlation in upper
  lower = list(continuous = "points"), # scatter in lower
  diag  = list(continuous = "densityDiag")
) +
  theme_bw()

p