# Please run 'Gradient_Boost_Model_submit.R' first.
library(caret)
library(tidyverse)
options(scipen=999)#Turn off scientific notation as global setting



# Step 1: Load the dataset
holdout_data <- read_csv(file = "/Users/rong/Desktop/Predictive_Analytics/Final/Team5_Final_Compitition_code/OceanCrest_holdout_noresponse.csv")

library(skimr)
skim(holdout_data)

# # Step 2: Convert Numerical Variables into Categorical
# holdout_data$arrival_date_year <- as.factor(holdout_data$arrival_date_year)
# holdout_data$arrival_date_month <- as.factor(holdout_data$arrival_date_month)
# holdout_data$arrival_date_week_number <- as.factor(holdout_data$arrival_date_week_number)
# holdout_data$arrival_date_day_of_month <- as.factor(holdout_data$arrival_date_day_of_month)

# Convert the variables to factors
holdout_data <- holdout_data %>%
  mutate(
        is_repeated_guest = as.factor(is_repeated_guest)
  )



# Step 4: Handle Missing Data
# Convert "NULL" strings to NA in character and factor columns only
holdout_data <- holdout_data %>%
  mutate(across(where(is.character), ~na_if(., "NULL"))) %>%
  mutate(across(where(is.factor), ~na_if(as.character(.), "NULL")))

#  4-1 Create indicator variables for variables with a lot of missing data
holdout_data$agent_missing <- is.na(holdout_data$agent)
holdout_data$company_missing <- is.na(holdout_data$company)


#  4-3 Mode Imputation for categorical variables
mode_impute <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
holdout_data$agent <- ifelse(is.na(holdout_data$agent), mode_impute(holdout_data$agent), holdout_data$agent)
holdout_data$country <- ifelse(is.na(holdout_data$country), mode_impute(holdout_data$country), holdout_data$country)

# 4-4 Drop columns with high missingness (>75%)
threshold <- 0.75
holdout_data <- holdout_data %>% select_if(~mean(is.na(.)) < threshold)

skim(holdout_data)




# Step 6: Feature Engineering
# 6.1: Create Total Stay Duration
holdout_data$total_stay <- holdout_data$stays_in_weekend_nights + holdout_data$stays_in_week_nights


# 6.3: Create Interaction Terms
holdout_data$interaction_weekend_week <- holdout_data$stays_in_weekend_nights * holdout_data$stays_in_week_nights
holdout_data$interaction_lead_totalstay <- holdout_data$lead_time * holdout_data$total_stay


# 6.4: Create Time-Based Features
# Creating seasonality variable based on 'arrival_date_month'
# 6.4.1 Create the season variable
month_to_num <- c(
  "January" = 1, "February" = 2, "March" = 3,
  "April" = 4, "May" = 5, "June" = 6,
  "July" = 7, "August" = 8, "September" = 9,
  "October" = 10, "November" = 11, "December" = 12
)

holdout_data$arrival_date_month_numeric <- month_to_num[holdout_data$arrival_date_month]

holdout_data$season <- cut(holdout_data$arrival_date_month_numeric, 
                              breaks = c(0, 3, 6, 9, 12), 
                              labels = c("Winter", "Spring", "Summer", "Fall"),
                              include.lowest = TRUE)


# Drop the 'arrival_date_day_of_month' column
holdout_data <- holdout_data %>% select(-arrival_date_day_of_month)


# Check the structure of the new variable
# table(holdout_data$day_in_month_category)

# Drop the original time columns: 'arrival_date_month' and 'arrival_date_day_of_month'
holdout_data <- holdout_data %>% 
  select(-arrival_date_month,-arrival_date_month_numeric)


# Step 7: Delete Irrelevant and Sparse Features 

####-----
# delete reservation_status_date
holdout_data <- holdout_data %>% select(-reservation_status_date,
                                        -reservation_status,
                                        -arrival_date_week_number,
                                        -agent)


skim(holdout_data)

# Create dummy variables using model.matrix() (excluding the intercept)

# Step 1: Extract the column names from OC_data_transformed to use as the reference
reference_columns <- colnames(OC_data_transformed)

# Step 2: Ensure the holdout data is transformed using the same factor levels as the training set
# (this step was covered in previous responses)
for (col in colnames(holdout_data)) {
  if (is.factor(OC_data[[col]]) && is.factor(holdout_data[[col]])) {
    holdout_data[[col]] <- factor(holdout_data[[col]], levels = levels(OC_data[[col]]))
  }
}

# Step 3: Create dummy variables for the holdout set
holdout_data_transformed <- model.matrix(~ . - 1, data = holdout_data)

# Convert the matrix to a data frame to allow dynamic column addition
holdout_data_transformed <- as.data.frame(holdout_data_transformed)

# Step 4: Align the holdout dataset with the reference columns from the training set
# Identify missing columns in the holdout set and add them with 0 values
missing_columns <- setdiff(reference_columns, colnames(holdout_data_transformed))

# Add missing columns with 0 values to the holdout set
for (col in missing_columns) {
  holdout_data_transformed[[col]] <- 0  # Use data frame column addition
}

# Step 5: Reorder columns in holdout_data_transformed to match the order of columns in OC_data_transformed
holdout_data_transformed <- holdout_data_transformed[, reference_columns]

# Step 6: Verify that both datasets now have the same structure
all(colnames(OC_data_transformed) == colnames(holdout_data_transformed))  # Should return TRUE

# (Optional) Convert back to matrix if required for modeling
holdout_data_transformed <- as.matrix(holdout_data_transformed)

# Step 7: Subset holdout data using only the important features selected in SHAP analysis
# Subset the holdout data to keep only the important features
holdout_data_reduced <- holdout_data_transformed[, valid_features]

# Step 8: Get the predicted probabilities for the holdout set using the reduced model
# Predict on the reduced holdout dataset
case_holdout_prob <- predict(model_gbm_reduced, holdout_data_reduced, type = "prob")

# Step 9: Combine the holdout data with the predicted probabilities
# Assuming we want to add the probabilities for the "1" class (cancellation)
case_holdout_scored <- cbind(holdout_data_reduced, case_holdout_prob$`1`)  # `1` is the class "canceled"

# Step 10: (Optional) Rename the probability column for clarity
colnames(case_holdout_scored)[ncol(case_holdout_scored)] <- "predicted_prob_canceled"

# Step 11: View the first few rows of the scored holdout data
head(case_holdout_scored)

# Step 12: (Optional) If needed, write the scored dataset to a CSV file
write.csv(case_holdout_scored, "/Users/rong/Desktop/Predictive_Analytics/Final/Team5_Final_Compitition_code/OC_holdout_scored.csv", row.names = FALSE)


