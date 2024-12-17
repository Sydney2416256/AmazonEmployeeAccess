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
library(themis)




# Load in Data
data <- vroom("STAT348/amazon/train.csv") 
testdata <- vroom("STAT348/amazon/test.csv") 

data$ACTION = as.factor(data$ACTION)






my_recipe <- recipe(ACTION ~. , data=data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_upsample(all_outcomes())
  # also step_upsample() and step_downsample()

# apply the recipe to your data
prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = testdata)




# prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet




log_reg_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")



wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_reg_mod)|> 
  fit(data=data)




## Make predictions
amazon_predictions <- predict(wf,
                              new_data=testdata,
                              type= "prob") 




kaggle_submission <- amazon_predictions %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


## Write out the file
vroom_write(x=kaggle_submission, file="./Amazon2.csv", delim=",")
