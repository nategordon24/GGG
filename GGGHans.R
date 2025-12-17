library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(kernlab)

### SVM
# import data
trainData <- vroom("train.csv")
testData <- vroom("test.csv")


#recipe
my_recipe <- recipe(type ~ ., data=trainData) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_mutate(hair465  = hair_length < 0.465) %>%
  #step_dummy(color)
  step_lencode_glm(color, outcome = vars(type)) #%>% #target encoding (must be 2-f
  #step_normalize(all_numeric_predictors()) #%>%
  #step_pca(all_predictors(), threshold = 0.99)



# SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")



# workflow
svmPoly_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

svmRadial_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)


# Grid of values to tune over
L = 5 # number of penalties and mixure
K = 5 # number of folds

grid_of_tuning_params_Poly <- grid_regular(cost(),
                                           degree(),
                                           levels = L)

grid_of_tuning_params_Radial <- grid_regular(cost(),
                                             rbf_sigma(),
                                             levels = L)


folds <- vfold_cv(trainData, v = K, repeats = 1)


# Run the CV
CV_results_Poly <- svmPoly_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params_Poly,
            metrics = metric_set(roc_auc))

CV_results_Radial <- svmRadial_wf %>%
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params_Radial,
            metrics = metric_set(roc_auc))


# Fine Best Tuning Parameters
bestTune_Poly <- CV_results_Poly %>%
  select_best(metric = "roc_auc")

bestTune_Radial <- CV_results_Radial %>%
  select_best(metric = "roc_auc")


# Finalize the workflow & fit it
final_wf_Poly <- svmPoly_wf %>%
  finalize_workflow(bestTune_Poly) %>%
  fit(data = trainData)

final_wf_Radial <- svmRadial_wf %>%
  finalize_workflow(bestTune_Radial) %>%
  fit(data = trainData)


# predict
my.pca.svmPoly.preds <- predict(final_wf_Poly, new_data = testData, type = 'class')
my.pca.svmRadial.preds <- predict(final_wf_Radial, new_data = testData, type = 'class')


# For submission
kaggle.ggg.svmPoly.preds <- my.pca.svmPoly.preds %>%
  bind_cols(testData) %>%
  mutate(type = .pred_class) %>%
  select(id, type)
  
kaggle.ggg.svmRadial.preds <- my.pca.svmRadial.preds %>%
  bind_cols(testData) %>%
  mutate(type = .pred_class) %>%
  select(id, type)


vroom_write(x=kaggle.ggg.svmPoly.preds, file="./kaggle_ggg_targethaircolor_svmPoly_preds.csv", delim=",")
vroom_write(x=kaggle.ggg.svmRadial.preds, file="./kaggle_ggg_targethaircolor_svmRadial_preds.csv", delim=",")

