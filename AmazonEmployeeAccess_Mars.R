
library(doParallel)

num_cores <- parallel::detectCores() #How many cores do I have?3
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)



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





mars <- mars(
  mode = "classification",
  engine = "earth",
  num_terms = NULL,
  prod_degree = NULL,
  prune_method = NULL
)








wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(mars)








##############################
amazon_predictions <- predict(wf,
                              new_data=testdata,
                              type= "prob") 


kaggle_submission <- amazon_predictions %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


## Write out the file
vroom_write(x=kaggle_submission, file="./Amazon_Mars.csv", delim=",")


stopCluster(cl)
