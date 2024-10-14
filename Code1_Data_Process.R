library(caret)
library(tidyverse)
options(scipen=999)#Turn off scientific notation as global setting

# Step 1: Load the dataset
OceanCrest_data <- read_csv(file = "/Users/rong/Desktop/Predictive_Analytics/week2/OceanCrestdata.csv")

library(skimr)
skim(OceanCrest_data)

# Step 2: Convert Numerical Variables into Categorical
OceanCrest_data$arrival_date_year <- as.factor(OceanCrest_data$arrival_date_year)
OceanCrest_data$arrival_date_month <- as.factor(OceanCrest_data$arrival_date_month)
OceanCrest_data$arrival_date_week_number <- as.factor(OceanCrest_data$arrival_date_week_number)
OceanCrest_data$arrival_date_day_of_month <- as.factor(OceanCrest_data$arrival_date_day_of_month)

# Convert the variables to factors
OceanCrest_data <- OceanCrest_data %>%
  mutate(
    is_canceled = as.factor(is_canceled),
    is_repeated_guest = as.factor(is_repeated_guest)
  )


# Step 2: Handle Duplicate rows and Columns
#   2-1 Drop the first column (index column)
OceanCrest_data <- OceanCrest_data[, -1]

#   2-2 We already know 'cases_info' is a duplicate of 'is_cancelled'
OceanCrest_data <- OceanCrest_data %>% select(-cases_info)

#   2-3 Check and delete Duplicate Rows:
duplicate_rows <- sum(duplicated(OceanCrest_data))
duplicate_rows
#   Remove duplicate rows
OceanCrest_data <- OceanCrest_data %>% distinct()

# Step 3: Correct Inconsistencies
# Flag negative values in 'adr' as missing value NA
OceanCrest_data$adr <- ifelse(OceanCrest_data$adr < 0, NA, OceanCrest_data$adr)

# skim(OceanCrest_data)

# Step 4: Handle Missing Data
# Convert "NULL" strings to NA in character and factor columns only
OceanCrest_data <- OceanCrest_data %>%
  mutate(across(where(is.character), ~na_if(., "NULL"))) %>%
  mutate(across(where(is.factor), ~na_if(as.character(.), "NULL")))

#  4-1 Create indicator variables for variables with a lot of missing data
OceanCrest_data$agent_missing <- is.na(OceanCrest_data$agent)
OceanCrest_data$company_missing <- is.na(OceanCrest_data$company)

#  4-2 Median Imputation for numeric variables
OceanCrest_data$children <- ifelse(is.na(OceanCrest_data$children), median(OceanCrest_data$children, na.rm = TRUE), OceanCrest_data$children)
OceanCrest_data$adr <- ifelse(is.na(OceanCrest_data$adr), median(OceanCrest_data$adr, na.rm = TRUE), OceanCrest_data$adr)

#  4-3 Mode Imputation for categorical variables
mode_impute <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
OceanCrest_data$agent <- ifelse(is.na(OceanCrest_data$agent), mode_impute(OceanCrest_data$agent), OceanCrest_data$agent)
OceanCrest_data$country <- ifelse(is.na(OceanCrest_data$country), mode_impute(OceanCrest_data$country), OceanCrest_data$country)

# 4-4 Drop columns with high missingness (>75%)
threshold <- 0.75
OceanCrest_data <- OceanCrest_data %>% select_if(~mean(is.na(.)) < threshold)

skim(OceanCrest_data)


#Step 5: Detect and Treat Outliers
# Calculate the maximum 'adr' value, ignoring NA values
max_adr_value <- max(OceanCrest_data$adr, na.rm = TRUE)

# Remove the row with the maximum value in the 'adr' column
OceanCrest_data <- OceanCrest_data %>%
  filter(adr != max_adr_value)

skim(OceanCrest_data)

# Step 6: Feature Engineering
# 6.1: Create Total Stay Duration
OceanCrest_data$total_stay <- OceanCrest_data$stays_in_weekend_nights + OceanCrest_data$stays_in_week_nights

# # 6.2: Create Polynomial Features
# OceanCrest_data$lead_time_squared <- OceanCrest_data$lead_time^2

# 6.3: Create Interaction Terms
OceanCrest_data$interaction_weekend_week <- OceanCrest_data$stays_in_weekend_nights * OceanCrest_data$stays_in_week_nights
OceanCrest_data$interaction_lead_totalstay <- OceanCrest_data$lead_time * OceanCrest_data$total_stay


# # Aggregate the data by week number to get cancellation rates per week
# weekly_cancellation_rate <- OceanCrest_data %>%
#   group_by(arrival_date_week_number) %>%
#   summarise(cancellations = sum(is_canceled),
#             total_bookings = n(),
#             cancellation_rate = mean(is_canceled))

# # Visualize the cancellation rate by week number
# library(ggplot2)
# ggplot(weekly_cancellation_rate, aes(x = as.numeric(arrival_date_week_number), y = cancellation_rate)) +
#   geom_line() +
#   labs(title = "Cancellation Rate by Week Number", x = "Week Number", y = "Cancellation Rate") +
#   theme_minimal()





# 6.4: Create Time-Based Features
# Creating seasonality variable based on 'arrival_date_month'
# 6.4.1 Create the season variable
month_to_num <- c(
  "January" = 1, "February" = 2, "March" = 3,
  "April" = 4, "May" = 5, "June" = 6,
  "July" = 7, "August" = 8, "September" = 9,
  "October" = 10, "November" = 11, "December" = 12
)

OceanCrest_data$arrival_date_month_numeric <- month_to_num[OceanCrest_data$arrival_date_month]

OceanCrest_data$season <- cut(OceanCrest_data$arrival_date_month_numeric, 
                              breaks = c(0, 3, 6, 9, 12), 
                              labels = c("Winter", "Spring", "Summer", "Fall"),
                              include.lowest = TRUE)

# # Calculate descriptive statistics for each day of the month
# summary_by_day <- OceanCrest_data %>%
#   group_by(arrival_date_day_of_month) %>%
#   summarise(
#     mean_cancellation = mean(is_canceled, na.rm = TRUE),
#     median_cancellation = median(is_canceled, na.rm = TRUE),
#     std_cancellation = sd(is_canceled, na.rm = TRUE)
#   )
# 
# # Print the summary statistics
# print(summary_by_day)

# Drop the 'arrival_date_day_of_month' column
OceanCrest_data <- OceanCrest_data %>% select(-arrival_date_day_of_month)

# Check the structure of the dataset to confirm the column is removed
#skim(OceanCrest_data)

# # 6.4.2 Create a new variable 'day_in_month_category' to categorize days
# OceanCrest_data$day_in_month_category <- cut(as.numeric(OceanCrest_data$arrival_date_day_of_month),
#                                              breaks = c(0, 10, 20, 31),
#                                              labels = c("Early", "Mid", "Late"),
#                                              include.lowest = TRUE)

# Check the structure of the new variable
# table(OceanCrest_data$day_in_month_category)

# Drop the original time columns: 'arrival_date_month' and 'arrival_date_day_of_month'
OceanCrest_data <- OceanCrest_data %>% 
  select(-arrival_date_month,-arrival_date_month_numeric)


#skim(OceanCrest_data)

# Step 7: Delete Irrelevant and Sparse Features

# 
# # Drop sparse features with too many unique categories
# # Identify sparse features with too many unique categories relative to the number of samples
# categorical_cols <- sapply(OceanCrest_data, is.factor) | sapply(OceanCrest_data, is.character)
# # Sparse if the number of unique values is greater than 50% of the number of rows
# sparse_cols <- sapply(OceanCrest_data[, categorical_cols], function(col) length(unique(col)) > 0.5 * nrow(OceanCrest_data))
# sparse_cols
# 
# OceanCrest_data <- OceanCrest_data[, !names(OceanCrest_data) %in% names(sparse_cols[sparse_cols])]
# skim(OceanCrest_data)

# 8.1: Standardize specific numerical variables
# Specify the columns to be standardized
# ####cols_to_standardize <- c("adr", "lead_time")

# Apply standardization only to the specified columns
# ###OceanCrest_data[cols_to_standardize] <- scale(OceanCrest_data[cols_to_standardize])

# #8.2: Discretize Continuous Variables
# #OceanCrest_data$total_stay_binned <- cut(OceanCrest_data$total_stay, breaks = 3, labels = c("Short", "Medium", "Long"))
# #skim(OceanCrest_data)

# # Step 9: Convert Categorical Variables into Dummy Variables
# 
# 
# # Exclude `reservation_status_date` and any other non-categorical columns
# categorical_cols <- sapply(OceanCrest_data, is.factor) | sapply(OceanCrest_data, is.character)
# 
# # Use model.matrix to create dummy variables, excluding `reservation_status_date`
# dummy_vars <- model.matrix(~ . - 1, data = OceanCrest_data[, categorical_cols])
# 
# # Convert the dummy variables into a data frame
# dummy_vars_df <- as.data.frame(dummy_vars)
# 
# # Bind the dummy variables back to the original dataset (excluding original categorical columns)
# OceanCrest_data <- cbind(OceanCrest_data[, !categorical_cols], dummy_vars_df)
# 
# # Check the structure of the final dataset
# skim(OceanCrest_data)
####-----
# delete reservation_status_date
OceanCrest_data <- OceanCrest_data %>% select(-reservation_status_date,
                                              -arrival_date_week_number,
                                              -agent)


skim(OceanCrest_data)

# Save the processed OceanCrest_data
write.csv(OceanCrest_data, "/Users/rong/Desktop/Predictive_Analytics/Final/OceanCrestdata_cleaned.csv", row.names = FALSE)

