# Load necessary libraries
library(caret)
library(tidyverse)
library(doParallel)
library(SHAPforxgboost)
library(pROC)
library(ROCR)

#Step 0 countinue to process with cleaned data
# Load the data
OC_data <- read.csv("/Users/rong/Desktop/Predictive_Analytics/Final/OceanCrestdata_cleaned.csv")


# Convert target variable to a factor
OC_data$is_canceled <- as.factor(OC_data$is_canceled)

# Create dummy variables using model.matrix() (excluding the intercept)
OC_data_transformed <- model.matrix(is_canceled ~ . - 1, data = OC_data)  # '-1' removes the intercept column

# Add back the target variable as a factor for training
OC_data_transformed <- as.data.frame(OC_data_transformed)
OC_data_transformed$is_canceled <- as.factor(as.character(OC_data$is_canceled))

#Step 1.  Split the data into training and testing sets
set.seed(99)
index <- createDataPartition(OC_data_transformed$is_canceled, p = .8, list = FALSE)
OC_train <- OC_data_transformed[index,]
OC_test <- OC_data_transformed[-index,]

# Set up parallel processing
num_cores <- detectCores() - 2
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Step 2.  Fit the model
#### 2 - 1 Full Model----
# Train the full XGBoost model using the correct factor outcome
set.seed(8)
model_gbm <- train(is_canceled ~ .,
                   data = OC_train,
                   method = "xgbTree",
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = expand.grid(
                     nrounds = 300,           # Optimal number of boosting rounds
                     eta = 0.02,              # Optimal learning rate
                     max_depth = 5,           # Optimal max tree depth
                     gamma = 0.1,             # Optimal gamma value
                     colsample_bytree = 1,    # Optimal column sampling ratio
                     min_child_weight = 1,    # Optimal min child weight
                     subsample = 0.7          # Optimal subsample ratio
                   ),
                   verbose = FALSE)


# plot(model_gbm)
# model_gbm$bestTune
# plot(varImp(model_gbm))


# Full model prediction and AUC calculation
pred_prob_full <- predict(model_gbm, OC_test, type = "prob")

# Calculate the AUC for full model
roc_obj_full <- roc(OC_test$is_canceled, as.numeric(pred_prob_full[, 2]))  # Assuming second column is class "1"
auc_value_full <- auc(roc_obj_full)
print(paste("Full model AUC:", auc_value_full))

# Plot ROC curve with color gradient using ROCR for Full Model
pred_full <- prediction(pred_prob_full[, 2], OC_test$is_canceled)  # Probabilities for class "1"
perf_full <- performance(pred_full, "tpr", "fpr")
plot(perf_full, colorize = TRUE, main = "ROC Curve with Color Gradient for Full Model")


####**SHAP analysis** to identify important features----
# 
Xdata <- as.matrix(select(OC_train, -is_canceled))

# Align feature names between model and SHAP
model_feature_names <- gsub("`", "", model_gbm$finalModel$feature_names)
model_gbm$finalModel$feature_names <- model_feature_names
Xdata <- Xdata[, model_gbm$finalModel$feature_names]

# Perform SHAP analysis
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# Calculate SHAP importance
shap_importance <- shap %>%
  group_by(variable) %>%
  summarise(Importance = mean(abs(value)))

# View the top features by SHAP importance and filter important features
shap_importance <- shap_importance %>% arrange(desc(Importance))
important_features <- shap_importance %>% filter(Importance > 0.01)
print(important_features)

# **Subset the data using only important features**
# Ensure that 'important_features$variable' matches with the column names in OC_train
valid_features <- intersect(important_features$variable, colnames(OC_train))
valid_features 
# Subset the training and test data using the valid features
OC_train_reduced <- OC_train[, c(valid_features, "is_canceled")]
OC_test_reduced <- OC_test[, valid_features]

#### 2 - 2 Reduced Model----

# ## fine-tune reduced model
# set.seed(8)
# tuneGrid_reduced <- expand.grid(
#   nrounds = c(100, 300),            # Number of trees (boosting rounds)
#   eta = c(0.01, 0.02),              # Learning rate
#   max_depth = c(3, 5),              # Maximum tree depth
#   gamma = c(0, 0.1),                # Minimum loss reduction for a split
#   colsample_bytree = c(0.7, 1),     # Subsample of columns
#   min_child_weight = c(1, 3),       # Minimum sum of instance weight (Hessian)
#   subsample = c(0.7, 1)             # Subsample ratio of training data
# )
# 
# # Train the reduced model with fine-tuning
# model_gbm_reduced_tuned <- train(
#   is_canceled ~ .,
#   data = OC_train_reduced,
#   method = "xgbTree",
#   trControl = trainControl(method = "cv", number = 5),
#   tuneGrid = tuneGrid_reduced,
#   verbose = FALSE
# )
# 
# 
# plot(model_gbm_reduced_tuned)
# 
# model_gbm_reduced_tuned$bestTune



# **Retrain the XGBoost model using the reduced feature set**
set.seed(8)
model_gbm_reduced <- train(is_canceled ~ .,
                           data = OC_train_reduced,
                           method = "xgbTree",
                           trControl = trainControl(method = "cv", number = 5),
                           tuneGrid = expand.grid(
                             nrounds = 300,
                             eta = 0.02,
                             max_depth = 5,
                             gamma = 0.1,
                             colsample_bytree = 1,
                             min_child_weight = 1,
                             subsample = 0.7),
                           verbose = FALSE)


#plot(model_gbm_reduced). ## Since tuning parameters are fixed, skip the plot
model_gbm_reduced$bestTune
plot(varImp(model_gbm_reduced))



# # Load necessary libraries
# library(caret)
# library(tidyverse)
# 
# # Extract variable importance from the reduced model
# importance_reduced <- varImp(model_gbm_reduced)
# 
# # Convert the variable importance to a data frame
# importance_df <- as.data.frame(importance_reduced$importance)
# importance_df <- rownames_to_column(importance_df, var = "Features")
# importance_df <- importance_df %>% arrange(desc(Overall))
# 
# # Highlight top 6 features
# importance_df <- importance_df %>%
#   mutate(Top6 = ifelse(Features %in% head(importance_df$Features, 6), "Top 6", "Other"))
# 
# # Create a ggplot similar to your example
# importance_plot <- ggplot(importance_df, aes(x = reorder(Features, Overall), y = Overall, fill = Top6)) +
#   geom_bar(stat = "identity", width = 0.7) +
#   coord_flip() +  # Flip to get horizontal bars
#   scale_fill_manual(values = c("Top 6" = "green3", "Other" = "skyblue")) +
#   labs(title = "Feature Importance in Predicting Cancellations",
#        x = "Features",
#        y = "Importance") +
#   theme_minimal() +
#   theme(legend.title = element_blank())
# 
# # Print the plot
# print(importance_plot)





# **Get predictions for the reduced model on the test set**
pred_prob_reduced <- predict(model_gbm_reduced, OC_test_reduced, type = "prob")

# **Calculate the AUC using pROC for the reduced model**
roc_obj_reduced <- roc(OC_test$is_canceled, as.numeric(pred_prob_reduced[, 2]))  # Assuming second column is class "1"
auc_value_reduced <- auc(roc_obj_reduced)
print(paste("Reduced model AUC:", auc_value_reduced))

# **Plot ROC curve with color gradient using ROCR for Reduced Model**
pred_reduced <- prediction(pred_prob_reduced[, 2], OC_test$is_canceled)
perf_reduced <- performance(pred_reduced, "tpr", "fpr")
plot(perf_reduced, colorize = TRUE, main = "ROC Curve with Color Gradient for Reduced Model")


####SHAP analysis on the reduced model----
#SHAP analysis on the reduced model
# Perform SHAP analysis on the reduced model using the reduced dataset
Xdata_reduced <- as.matrix(select(OC_train_reduced, -is_canceled))  # Use the preprocessed data without the target

# Ensure feature names match between the reduced dataset and the reduced model
model_feature_names_reduced <- gsub("`", "", model_gbm_reduced$finalModel$feature_names)
model_gbm_reduced$finalModel$feature_names <- model_feature_names_reduced

# Subset Xdata_reduced to match the feature names used in the reduced model
Xdata_reduced <- Xdata_reduced[, model_gbm_reduced$finalModel$feature_names]

# Perform SHAP analysis for the reduced model
shap_reduced <- shap.prep(model_gbm_reduced$finalModel, X_train = Xdata_reduced)

# SHAP importance summary for the reduced model
shap.plot.summary(shap_reduced)

# Calculate SHAP importance as a data frame
shap_importance_reduced <- shap_reduced %>%
  group_by(variable) %>%
  summarise(Importance = mean(abs(value)))

# View the top features by SHAP importance in the reduced model
shap_importance_reduced <- shap_importance_reduced %>% arrange(desc(Importance))
print(shap_importance_reduced)



# SHAP dependence plots for the top 6 important features
top6_features <- head(shap_importance_reduced$variable, 6)
for (feature in top6_features) {
  p <- shap.plot.dependence(shap_reduced, x = feature, color_feature = "auto") + ggtitle(paste("Dependence plot for", feature))
  print(p)
}



# **Stop parallel processing**
stopCluster(cl)

####Export the Subset Dataset ----

# Subset the training and test datasets to keep only the important features and target variable
OC_train_reduced <- OC_train[, c(valid_features, "is_canceled")]
OC_test_reduced <- OC_test[, c(valid_features, "is_canceled")]


# Combine the two datasets (train and test)
OC_combined <- rbind(OC_train_reduced, OC_test_reduced)

# Print the structure of the combined dataset
print("Combined dataset (train and test):")
str(OC_combined)

# Optionally, save the combined dataset to a CSV file
write.csv(OC_combined, "OC_combined_reduced.csv", row.names = FALSE)



####compute the metrics like accuracy, precision, recall, and F1 score.----
# Use existing predicted probabilities from the full and reduced models
# (pred_prob_full and pred_prob_reduced)

# Convert probabilities to binary class labels (0 or 1) using a 0.5 threshold
pred_class_full <- ifelse(pred_prob_full[, 2] > 0.5, 1, 0)
pred_class_reduced <- ifelse(pred_prob_reduced[, 2] > 0.5, 1, 0)

# Calculate confusion matrix for both models using caret's confusionMatrix function
conf_matrix_full <- confusionMatrix(factor(pred_class_full), OC_test$is_canceled)
conf_matrix_reduced <- confusionMatrix(factor(pred_class_reduced), OC_test$is_canceled)

# Print the confusion matrix for both models
print(conf_matrix_full)
print(conf_matrix_reduced)

# Extract evaluation metrics from the confusion matrix for the full model
accuracy_full <- conf_matrix_full$overall['Accuracy']
precision_full <- conf_matrix_full$byClass['Pos Pred Value']  # Precision
recall_full <- conf_matrix_full$byClass['Sensitivity']  # Recall
f1_score_full <- 2 * ((precision_full * recall_full) / (precision_full + recall_full))

# Print metrics for the full model
print(paste("Full Model Accuracy:", accuracy_full))
print(paste("Full Model Precision:", precision_full))
print(paste("Full Model Recall:", recall_full))
print(paste("Full Model F1 Score:", f1_score_full))

# Extract evaluation metrics from the confusion matrix for the reduced model
accuracy_reduced <- conf_matrix_reduced$overall['Accuracy']
precision_reduced <- conf_matrix_reduced$byClass['Pos Pred Value']  # Precision
recall_reduced <- conf_matrix_reduced$byClass['Sensitivity']  # Recall
f1_score_reduced <- 2 * ((precision_reduced * recall_reduced) / (precision_reduced + recall_reduced))

# Print metrics for the reduced model
print(paste("Reduced Model Accuracy:", accuracy_reduced))
print(paste("Reduced Model Precision:", precision_reduced))
print(paste("Reduced Model Recall:", recall_reduced))
print(paste("Reduced Model F1 Score:", f1_score_reduced))


