



# Load in Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)
library(tidymodels)
library(dplyr)
library(poissonreg)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(embed)
library(kernlab)


# Load in Data
data <- vroom("STAT348/amazon/train.csv") 
testdata <- vroom("STAT348/amazon/test.csv") 

data$ACTION = as.factor(data$ACTION)






my_recipe <- recipe(ACTION ~. , data=data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_predictors(), threshold = .001) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold= .9) 





prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet





svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")









wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmLinear)






## Grid of values to tune over
tuning_grid <- grid_regular(cost(),
                            levels = 5) 

## Split data for CV
folds <- vfold_cv(data, v = 5, repeats=1)






##################################
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL


bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_Linear <- wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=data)






##############################
amazon_predictions <- predict(final_Linear,
                              new_data=testdata,
                              type= "prob") 


kaggle_submission <- amazon_predictions %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


## Write out the file
vroom_write(x=kaggle_submission, file="./Amazon_SVM_Linear.csv", delim=",")



